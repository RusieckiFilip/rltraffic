"""Machine B (docs/notes/MACHINE_B.md): two checkpoints of one training cell, compared tensor by tensor.

Run with the project's venv (torch, numpy):  .venv/bin/python -I machine_b_compare.py <checkpoint A> <checkpoint B>

Each checkpoint is a P9.3 training checkpoint, ``<root>/p9_3/checkpoints/<cell>.pt``; its training record,
``<root>/p9_3/train/<cell>.json``, supplies the loss. A is the reference (the laptop), B the machine compared with it.

For every tensor of the model it prints the number of elements that differ, the largest absolute difference, that
difference relative to the tensor's largest magnitude in A, and the largest elementwise difference in float32 ulps
(the spacing of float32 at the larger of the two values). It then compares the recorded losses and the checkpoints'
other contents (configuration, normalisation statistics, prompts, mask), which come from the corpus, not from training.

The bound, the author's "floating-point rounding size" (2026-10-29), fixed before any comparison was computed:
  * every tensor's largest absolute difference is at most BOUND_ABS = 1e-5, a tenth of one AdamW step. These cells
    train at learning rate 1e-4 with no warmup (2 steps, warmup 1), so each step moves nearly every weight by about
    1e-4; any difference in what was computed (data, code, gradient direction) moves weights by up to that much per
    step, while rounding moves them by a few ulps (about 1e-9 to 1e-7 at these magnitudes);
  * the recorded losses agree to a relative BOUND_LOSS = 1e-5 (about 85 ulps at the loss's size, 2.07).
The verdict line says whether both hold. Loading uses ``weights_only=True``: nothing in a checkpoint is executed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BOUND_ABS = 1e-5
BOUND_LOSS = 1e-5
#: The checkpoint keys that come from the corpus and the configuration, not from training: equal on any machine.
FROM_THE_CORPUS = ("format_version", "config", "target_rtg", "rtg_scale", "normalise", "scenario_id", "stats",
                   "intersection_ids", "spatial_mask")


def record_of(checkpoint: Path) -> dict:
    """The training record beside *checkpoint*: ``.../p9_3/checkpoints/X.pt`` -> ``.../p9_3/train/X.json``."""
    return json.loads((checkpoint.parent.parent / "train" / (checkpoint.stem + ".json")).read_text())


def load(path: Path) -> dict:
    """A checkpoint, unpickled with nothing executed: its provenance's torch version (a ``str``) the one class allowed."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def compare(path_a: Path, path_b: Path) -> int:
    a, b = load(path_a), load(path_b)
    model_a, model_b = a["model"], b["model"]
    if sorted(model_a) != sorted(model_b):
        print(f"DIFFERENT TENSORS: only in A {sorted(set(model_a) - set(model_b))}, only in B {sorted(set(model_b) - set(model_a))}")
        return 1
    print(f"A: {path_a}\nB: {path_b}")
    print(f"{'tensor':58s} {'shape':>16s} {'differ':>14s} {'max abs':>10s} {'max rel':>10s} {'max ulps':>9s}")
    worst_abs = worst_rel = worst_ulps = 0.0
    differing = elements = changed = 0
    for key in sorted(model_a):
        ta, tb = model_a[key], model_b[key]
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            print(f"{key}: dtype or shape differs ({ta.dtype} {tuple(ta.shape)} against {tb.dtype} {tuple(tb.shape)})")
            return 1
        va = ta.detach().contiguous().numpy().astype(np.float64).ravel()
        vb = tb.detach().contiguous().numpy().astype(np.float64).ravel()
        diff = np.abs(va - vb)
        n = int(np.count_nonzero(diff))
        largest_a = float(np.max(np.abs(va))) if va.size else 0.0
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = max_abs / largest_a if largest_a > 0 else (0.0 if max_abs == 0 else float("inf"))
        spacing = np.spacing(np.maximum(np.abs(va), np.abs(vb)).astype(np.float32)).astype(np.float64)
        max_ulps = float(np.max(diff / spacing)) if n else 0.0
        worst_abs, worst_rel, worst_ulps = max(worst_abs, max_abs), max(worst_rel, max_rel), max(worst_ulps, max_ulps)
        elements += va.size
        changed += n
        differing += 1 if n else 0
        print(f"{key:58s} {str(tuple(ta.shape)):>16s} {n:>6d}/{va.size:<7d} {max_abs:10.3e} {max_rel:10.3e} {max_ulps:9.1f}")
    print(f"tensors: {len(model_a)}, of which {differing} differ; elements: {elements}, of which {changed} differ")
    print(f"largest absolute difference {worst_abs:.3e} (bound {BOUND_ABS:.0e}); largest relative {worst_rel:.3e}; "
          f"largest {worst_ulps:.1f} ulps")

    loss_a, loss_b = float(record_of(path_a)["final_loss"]), float(record_of(path_b)["final_loss"])
    loss_rel = abs(loss_a - loss_b) / abs(loss_a)
    print(f"recorded loss: A {loss_a!r}, B {loss_b!r}, relative difference {loss_rel:.3e} (bound {BOUND_LOSS:.0e})")

    same = [key for key in FROM_THE_CORPUS if json.dumps(a.get(key), sort_keys=True) == json.dumps(b.get(key), sort_keys=True)]
    other = [key for key in FROM_THE_CORPUS if key not in same]
    print(f"from the corpus and configuration: {len(same)} of {len(FROM_THE_CORPUS)} entries equal"
          + (f"; DIFFERENT: {other}" if other else ""))

    within = worst_abs <= BOUND_ABS and loss_rel <= BOUND_LOSS and not other
    print(f"VERDICT: {'within' if within else 'BEYOND'} the rounding bound")
    return 0 if within else 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    return compare(Path(argv[0]), Path(argv[1]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
