"""Amendment B, B2: does any C8 change touch a path the pinned pre-flight timed?  Compares the source of every timed
function at the pre-flight's commit (a71a72c) and at HEAD, and prints the two stage functions' diffs.
usage (from the task worktree): python docs/returns/P5.2b_evidence/c8/timed_paths_unchanged.py"""
import ast
import difflib
import subprocess


def source(commit: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


def functions(text: str) -> dict[str, str]:
    lines = text.splitlines()
    return {node.name: "\n".join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)}


TIMED = ["training_inputs", "train_seeds", "p5_2_evaluation_argv", "narrowed_held_out_draws", "run_p5_2_evaluation",
         "rederivation_cells", "committed_from_eval", "_probe_roots", "resolved_checkpoints", "run_rederivation",
         "preflight_estimate", "timeouts_from_preflight", "_compare", "_regime", "_canary_at_speed", "_canary"]
old, new = functions(source("a71a72c", "offline/iql_correction.py")), functions(source("HEAD", "offline/iql_correction.py"))
for name in TIMED:
    print(f"offline/iql_correction.py:{name:26s} {'unchanged' if old[name] == new[name] else 'CHANGED'}")
o, n = functions(source("a71a72c", "offline/tier_sweep.py")), functions(source("HEAD", "offline/tier_sweep.py"))
for name in ("iql_transition_table", "_run_train_baselines", "_run_evaluate", "main"):
    print(f"offline/tier_sweep.py:{name:26s} {'unchanged' if o[name] == n[name] else 'CHANGED'}")
for name in ("preflight", "train_stage"):
    print("".join(difflib.unified_diff(old[name].splitlines(True), new[name].splitlines(True), f"a71a72c {name}",
                                       f"HEAD {name}", n=0)))
