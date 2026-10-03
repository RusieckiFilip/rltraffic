"""G1 smoke of ONE latency cell (coordinator's check, BRIEF_43 gate G1): build the row's env for draw 1000 and its agent
through the row's own factory exactly as run_row does, then report WHERE the parameters live and WHICH context length
the DT runs at, and roll one episode of 360 decisions. No timing value and no episode quantity is printed or kept.

usage: python g1_cell.py <work_tree> <row_id> <device>
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

WORK_TREE, ROW_ID, DEVICE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, WORK_TREE)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import offline.compute_latency as cl  # noqa: E402

assert cl.__file__ == f"{WORK_TREE}/offline/compute_latency.py", cl.__file__

MAIN = Path("/home/filip/rltraffic")
OUT = {"row": ROW_ID, "device": DEVICE}


def walk(obj, depth, seen, modules, tensors, configs):
    if id(obj) in seen or depth > 4:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        modules.append(obj)
        return
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return
    if hasattr(obj, "context_length") and not isinstance(obj, type):
        try:
            configs.append(int(getattr(obj, "context_length")))
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            walk(item, depth + 1, seen, modules, tensors, configs)
    elif isinstance(obj, dict):
        for item in obj.values():
            walk(item, depth + 1, seen, modules, tensors, configs)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type) and type(obj).__module__.split(".")[0] in (
        "agent", "offline", "torch", "experiments", "algorithms", "__main__",
    ):
        for item in vars(obj).values():
            walk(item, depth + 1, seen, modules, tensors, configs)


row = cl.row_by_id(ROW_ID)
OUT["regime"] = cl.configure_regime(DEVICE)["torch_num_threads"]
settings, configs = cl._evaluation_inputs(row, MAIN / "datasets_v11", MAIN / "scenarios" / "draws")
factory, prompt, name = cl._row_factory(
    row, DEVICE, output_root=MAIN / "output", corpus_root=MAIN / "datasets_v11",
    data_dir=Path(WORK_TREE) / "docs" / "data", first_config=configs[cl.TIMING_DRAWS[0]],
)
OUT["factory"] = name.split(";")[0]
env = cl._make_env(row, settings, configs[cl.TIMING_DRAWS[0]])
choose = factory(env)
roots = [cell.cell_contents for cell in (choose.__closure__ or ())]
modules, tensors, ks = [], [], []
seen: set = set()
for root in roots:
    walk(root, 0, seen, modules, tensors, ks)
param_devices = sorted({p.device.type for m in modules for p in m.parameters()})
buffer_devices = sorted({b.device.type for m in modules for b in m.buffers()})
OUT["n_modules"] = len(modules)
OUT["n_params_seen"] = sum(p.numel() for m in {id(m): m for m in modules}.values() for p in m.parameters())
OUT["param_devices"] = param_devices
OUT["buffer_devices"] = buffer_devices
OUT["loose_tensor_devices"] = sorted({t.device.type for t in tensors})
OUT["context_lengths_seen"] = sorted(set(ks))
if row.checkpoint is not None and row.kind in ("dt", "spatial_dt", "spatial_dt_targets"):
    payload = torch.load(MAIN / row.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    OUT["checkpoint_context_length"] = payload.get("context_length", cfg.get("context_length"))
sync = cl._cuda_sync if DEVICE == "cuda" else None
decisions = cl.time_episode(env, choose, engine_seed=cl.ENGINE_SEED, sync=sync)
OUT["n_decisions"] = len(decisions)
del decisions
OUT["cuda_initialized"] = bool(torch.cuda.is_initialized())
OUT["cuda_max_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_initialized() else 0.0
print("G1CELL " + json.dumps(OUT, sort_keys=True), flush=True)
os._exit(0)
