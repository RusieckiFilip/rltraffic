"""G3: run the reviewed builder (from the run worktree at 5c8a33a) into the coordinator's scratchpad -- read-only use."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
WT = "/home/filip/rltraffic-p82-run"
sys.path.insert(0, WT)
import offline.compute_table as ct
assert ct.__file__ == f"{WT}/offline/compute_table.py", ct.__file__
t = time.monotonic()
code = ct.main(["build", "--output-root", "/home/filip/rltraffic/output", "--corpus-root", "/home/filip/rltraffic/datasets_v11",
                "--latency-dir", "/home/filip/rltraffic/output/p8_2/latency/20261004T194421Z",
                "--manifest", "/home/filip/rltraffic/output/SHA256SUMS_p8_2.txt", "--out", sys.argv[1]])
print("exit", code, "seconds", round(time.monotonic() - t, 1))
