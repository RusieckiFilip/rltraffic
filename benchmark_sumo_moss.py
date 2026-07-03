#!/usr/bin/env python3
"""SUMO vs MOSS Performance Benchmark

Runs N full episodes sequentially (SUMO first, then MOSS) and reports:
  - per-episode wall time, step speed, sim throughput
  - per-episode traffic metrics (avg travel time, vehicles completed, pressure)
  - step-level timing breakdown (simulate vs Python overhead)
  - final side-by-side comparison table

Usage:
    python3 benchmark_sumo_moss.py [options]

    --episodes N    episodes per backend        (default: 3)
    --steps    N    max env steps per episode   (default: 300  = 3000 sim-s)
    --delta    N    sim-seconds per env step    (default: 10)
    --warmup   N    MOSS GPU warm-up steps      (default: 50)
    --device   N    CUDA device index           (default: 0)
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from types import MethodType
from datetime import datetime
from typing import Any

# Project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# SUMO bootstrap
# eclipse-sumo (pip) ships the binary inside the package; add it to PATH and
# set SUMO_HOME so libsumo can find its data files.
try:
    import sumo as _sumo_pkg  # from eclipse-sumo pip wheel
    _sumo_home = _sumo_pkg.SUMO_HOME
    os.environ.setdefault("SUMO_HOME", _sumo_home)
    _bin = os.path.join(_sumo_home, "bin")
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + ":" + os.environ.get("PATH", "")
    _tools = os.path.join(_sumo_home, "tools")
    if os.path.isdir(_tools) and _tools not in sys.path:
        sys.path.insert(0, _tools)
except ImportError:
    # Fall back to system apt install (Ubuntu PPA)
    _sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    _tools = os.path.join(_sumo_home, "tools")
    if os.path.isdir(_tools) and _tools not in sys.path:
        sys.path.insert(0, _tools)

# Scenario paths
SCENARIOS = {
    "bb5b": {
        "label": "BB5B",
        "sumo_cfg": os.path.join(_HERE, "scenarios", "bb5b", "BB5B.sumocfg"),
        "moss_dir": os.path.join(_HERE, "moss_converted", "bb5b_mirrored"),
        "moss_start_step": 25200,
    },
    "hangzhou_4x4": {
        "label": "Hangzhou 4x4 Gudang",
        "sumo_cfg": os.path.join(
            _HERE,
            "scenarios",
            "hangzhou_4x4_gudang_18041610_1h",
            "hangzhou_4x4_gudang_18041610_1h.sumocfg",
        ),
        "moss_dir": os.path.join(_HERE, "moss_converted", "hangzhou_4x4_gudang"),
        "moss_start_step": 0,
    },
    "manhattan_28x7": {
        "label": "Manhattan 28x7",
        "sumo_cfg": os.path.join(
            _HERE,
            "scenarios",
            "manhattan_28x7",
            "manhattan_28x7.sumocfg",
        ),
        "moss_dir": os.path.join(_HERE, "scenarios", "manhatan"),
        "moss_start_step": 0,
    },
}

METRICS = [
    "average_travel_time",
    "average_intersection_pressure",
    "waiting_time_all_vehicles_for_the_last_time_step_in_simulation",
    "count_of_vehicles_completing_journey",
]

# CLI
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=3,   help="Episodes per backend")
    p.add_argument("--steps",    type=int, default=300, help="Max env steps per episode")
    p.add_argument("--delta",    type=int, default=10,  help="Sim-seconds per env step")
    p.add_argument("--warmup",   type=int, default=50,  help="MOSS warm-up steps (discarded)")
    p.add_argument("--device",   type=int, default=0,   help="CUDA device index for MOSS")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="bb5b",
                   help="Scenario to benchmark")
    p.add_argument("--moss-dir", default=None,
                   help="Override the scenario's MOSS map/person directory")
    p.add_argument("--policy", choices=["idqn", "fixed", "native"], default="idqn",
                   help="Controller policy: IDQN, constant phase action, or backend-native TL program")
    p.add_argument("--fixed-action", type=int, default=0,
                   help="Phase/action index used when --policy fixed")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed for IDQN")
    p.add_argument("--no-learn", action="store_true", help="Use IDQN actions without replay updates")
    p.add_argument("--eval",     action="store_true", help="Disable IDQN exploration")
    return p.parse_args()


def _scenario_paths(name: str) -> dict[str, Any]:
    cfg = dict(SCENARIOS[name])
    cfg["moss_map"] = os.path.join(cfg["moss_dir"], "map.pb")
    cfg["moss_person"] = os.path.join(cfg["moss_dir"], "person.pb")
    return cfg


def _apply_path_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.moss_dir:
        cfg["moss_dir"] = os.path.abspath(args.moss_dir)
        cfg["moss_map"] = os.path.join(cfg["moss_dir"], "map.pb")
        cfg["moss_person"] = os.path.join(cfg["moss_dir"], "person.pb")


# System info
def _gpu_info() -> str:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        rows = []
        for line in raw:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                idx, name, cc, mem = parts
                rows.append(f"[{idx}] {name}  CC {cc}  {int(float(mem))//1024} GB VRAM")
        return "\n         ".join(rows) if rows else "unavailable"
    except Exception:
        return "unavailable (is nvidia-smi in PATH?)"


def _cpu_info() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                name = line.split(":", 1)[1].strip()
                cores = os.cpu_count() or "?"
                return f"{name}  ({cores} logical cores)"
    except Exception:
        pass
    import platform
    return platform.processor() or "unknown"


# Helpers
_W = 70   # output width

def _sep(char="="): print(char * _W)
def _hsep():        print("-" * _W)


def _reward_scalar(reward: Any) -> float:
    try:
        return float(reward)
    except (TypeError, ValueError):
        try:
            import numpy as np

            return float(np.asarray(reward, dtype=np.float32).mean())
        except Exception:
            return 0.0


class FixedActionPolicy:
    def __init__(self, env: Any, fixed_action: int) -> None:
        self.intersections = list(getattr(env, "unwrapped", env).intersections)
        nvec = getattr(env.action_space, "nvec", None)
        if nvec is None:
            nvec = [getattr(env.action_space, "n", 1)]
        self.action_counts = [int(n) for n in list(nvec)]
        self.fixed_action = int(fixed_action)

    def act(self, info: dict[str, Any], explore: bool = False) -> list[int]:
        _ = info, explore
        return [
            min(max(self.fixed_action, 0), max(n_actions - 1, 0))
            for n_actions in self.action_counts
        ]

    def observe(
        self,
        next_info: dict[str, Any],
        reward: Any,
        terminated: bool,
        truncated: bool = False,
    ) -> dict[str, float]:
        _ = next_info, reward, terminated, truncated
        return {}


class NativePolicy(FixedActionPolicy):
    def __init__(self, env: Any) -> None:
        super().__init__(env, 0)


def _make_agent(env: Any, policy: str, fixed_action: int, seed: int) -> Any:
    if policy == "native":
        return NativePolicy(env)
    if policy == "fixed":
        return FixedActionPolicy(env, fixed_action)

    from agent.DQNAgent import IDQNAgent
    return IDQNAgent(env, seed=seed)


def _policy_label(policy: str, explore: bool, learn: bool, fixed_action: int) -> str:
    if policy == "native":
        return "NativeTrafficLightProgram"
    if policy == "fixed":
        return f"FixedActionPolicy  action={fixed_action}"
    return f"IDQNAgent  explore={explore}  learn={learn}"


def _run_episode(
    env: Any,
    agent: Any,
    max_steps: int,
    learn: bool,
    explore: bool,
) -> tuple[list[float], list[float], list[float], dict]:
    """
    Reset and run one episode.

    Returns
    -------
    step_times_ms : list[float]
        Wall time of each env.step() call in ms.
    sim_times_ms : list[float]
        Wall time of _simulate() only, in ms (via lightweight monkey-patch).
    rewards : list[float]
        Scalar reward of each step.
    final_metrics : dict
        info["metrics"] from the last step.
    """
    sim_times: list[float] = []
    _orig_simulate = env._simulate

    def _timed_simulate(self, n):
        t0 = time.perf_counter()
        _orig_simulate(n)
        sim_times.append((time.perf_counter() - t0) * 1000.0)

    env._simulate = MethodType(_timed_simulate, env)
    try:
        info = env.reset()
        step_times: list[float] = []
        rewards: list[float] = []
        final_metrics: dict = {}

        for _ in range(max_steps):
            t0 = time.perf_counter()
            action = agent.act(info, explore=explore)
            reward, terminated, truncated, next_info = env.step(action)
            rewards.append(_reward_scalar(reward))
            if learn:
                agent.observe(next_info, reward, bool(terminated), bool(truncated))
            step_times.append((time.perf_counter() - t0) * 1000.0)
            info = next_info
            final_metrics = info.get("metrics", {})
            if terminated or truncated:
                break
    finally:
        env._simulate = _orig_simulate

    return step_times, sim_times, rewards, final_metrics


def _ep_report(ep: int, n_eps: int, step_ms: list[float], sim_ms: list[float],
               rewards: list[float], metrics: dict, delta: int) -> None:
    ms    = statistics.mean(step_ms)
    std   = statistics.stdev(step_ms) if len(step_ms) > 1 else 0.0
    sps   = 1000.0 / ms
    s_ms  = statistics.mean(sim_ms) if sim_ms else 0.0
    py_ms = ms - s_ms
    steps = len(step_ms)

    tt       = metrics.get("average_travel_time", 0.0)
    compl    = int(metrics.get("count_of_vehicles_completing_journey", 0))
    pressure = metrics.get("average_intersection_pressure", 0.0)
    wait_now = metrics.get("waiting_time_all_vehicles_for_the_last_time_step_in_simulation", 0.0)
    ep_return = sum(rewards)
    mean_reward = statistics.mean(rewards) if rewards else 0.0

    print(f"  Episode {ep}/{n_eps}")
    _hsep()
    print(f"    Steps completed  : {steps}  ({steps * delta:,} simulated seconds)")
    print(f"    Reward           : return {ep_return:.2f}  |  {mean_reward:.3f} / step")
    print(f"    Wall time        : {steps * ms / 1000:.1f} s")
    print(f"    Speed            : {ms:.1f} +/- {std:.1f} ms/step"
          f"   |  {sps:.2f} steps/s   |  {sps * delta:.0f} sim-s/real-s")
    if sim_ms:
        py_pct = py_ms / ms * 100 if ms > 0 else 0
        print(f"    Breakdown        : {s_ms:.1f} ms simulate"
              f"  +  {py_ms:.1f} ms Python wrapper ({py_pct:.1f}%)")
    print(f"    Traffic at end   :")
    print(f"      avg travel time      {tt:.1f} s")
    print(f"      vehicles completed   {compl}")
    print(f"      avg pressure         {pressure:.3f} veh/intersection")
    print(f"      total active wait    {wait_now:.0f} s")
    print()


def _summarise(label: str,
               all_step: list[float], all_sim: list[float],
               all_rewards: list[float], all_metrics: list[dict], delta: int) -> dict:
    ms   = statistics.mean(all_step)
    std  = statistics.stdev(all_step) if len(all_step) > 1 else 0.0
    sps  = 1000.0 / ms
    s_ms = statistics.mean(all_sim) if all_sim else 0.0
    py_ms = ms - s_ms
    last = all_metrics[-1]
    total_return = sum(all_rewards)
    reward_step = statistics.mean(all_rewards) if all_rewards else 0.0

    result = {
        "ms": ms, "std": std, "sps": sps, "sim_sps": sps * delta,
        "sim_ms": s_ms, "py_ms": py_ms,
        "return": total_return, "reward_step": reward_step,
        "tt":       last.get("average_travel_time", 0.0),
        "compl":    int(last.get("count_of_vehicles_completing_journey", 0)),
        "pressure": last.get("average_intersection_pressure", 0.0),
    }
    _hsep()
    print(f"  {label} - all-episode aggregate")
    print(f"    return      : {total_return:.2f} total  |  {reward_step:.3f} reward/step")
    print(f"    ms / step   : {ms:.1f} +/- {std:.1f}")
    print(f"    steps / s   : {sps:.2f}   ({sps * delta:.0f} sim-s / real-s)")
    if all_sim:
        print(f"    simulate    : {s_ms:.1f} ms/step  "
              f"Python overhead: {py_ms:.1f} ms/step ({py_ms/ms*100:.1f}%)")
    print(f"    Traffic (last episode): "
          f"att={result['tt']:.0f}s  done={result['compl']}  press={result['pressure']:.3f}")
    print()
    return result


# SUMO
def bench_sumo(
    n_eps: int,
    max_steps: int,
    delta: int,
    scenario: dict[str, Any],
    *,
    seed: int,
    learn: bool,
    explore: bool,
    policy: str,
    fixed_action: int,
) -> dict | None:
    _sep()
    print("  1 / 2  SUMO  -  libsumo, CPU-based")
    _sep()

    sumo_cfg = scenario["sumo_cfg"]
    if not os.path.exists(sumo_cfg):
        print(f"  [SKIP] sumocfg not found: {sumo_cfg}")
        print()
        return None

    try:
        from envs.sumo_env import SumoEnv
        from envs.phase_control import AcyclicPhases
    except ImportError as exc:
        print(f"  [SKIP] import error: {exc}")
        print()
        return None

    if policy == "native":
        print("  [SKIP] SumoEnv does not support running the native TL program")
        print()
        return None

    t0 = time.perf_counter()
    try:
        env = SumoEnv(
            sumocfg_path=sumo_cfg,
            max_steps=max_steps,
            delta_time=delta,
            global_reward_fn="queue_length",
            phase_control_cls=AcyclicPhases,
            metrics=METRICS,
            libsumo=True,
        )
    except TypeError:
        # A TypeError means this script drifted from the env constructor
        # signature — that is a bug to fix, not a scenario to skip.
        raise
    except Exception as exc:
        print(f"  [SKIP] init failed: {exc}")
        print()
        return None

    t_init = time.perf_counter() - t0
    n_ix = len(env.intersections)
    print(f"  Init           : {t_init:.2f} s")
    print(f"  Intersections  : {n_ix}")
    print(f"  Policy         : {_policy_label(policy, explore, learn, fixed_action)}")
    print(f"  Config         : {os.path.basename(sumo_cfg)}")
    print(f"  Max steps/ep   : {max_steps}  ({max_steps * delta:,} sim-s)")
    print()

    agent = _make_agent(env, policy, fixed_action, seed)
    all_step: list[float] = []
    all_sim:  list[float] = []
    all_rewards: list[float] = []
    all_mets: list[dict]  = []

    for ep in range(1, n_eps + 1):
        step_ms, sim_ms, rewards, mets = _run_episode(env, agent, max_steps, learn, explore)
        all_step.extend(step_ms)
        all_sim.extend(sim_ms)
        all_rewards.extend(rewards)
        all_mets.append(mets)
        _ep_report(ep, n_eps, step_ms, sim_ms, rewards, mets, delta)

    env.close()
    return _summarise("SUMO", all_step, all_sim, all_rewards, all_mets, delta)


# MOSS
def bench_moss(n_eps: int, max_steps: int, delta: int,
               warmup: int, device: int, scenario: dict[str, Any], *,
               seed: int, learn: bool, explore: bool,
               policy: str, fixed_action: int) -> dict | None:
    _sep()
    print("  2 / 2  MOSS  -  GPU-accelerated")
    _sep()

    moss_map = scenario["moss_map"]
    moss_person = scenario["moss_person"]
    for path in (moss_map, moss_person):
        if not os.path.exists(path):
            print(f"  [SKIP] file not found: {path}")
            print()
            return None

    try:
        from envs.moss_env import MossEnv
        from envs.phase_control import AcyclicPhases
    except ImportError as exc:
        print(f"  [SKIP] import error: {exc}")
        print()
        return None

    print("  Note: first init compiles GPU kernels - expect 60-120 s")
    t0 = time.perf_counter()
    try:
        env = MossEnv(
            map_file=moss_map,
            person_file=moss_person,
            max_steps=max_steps,
            delta_time=delta,
            global_reward_fn="queue_length",
            phase_control_cls=AcyclicPhases,
            metrics=METRICS,
            device=device,
            start_step=scenario["moss_start_step"],
            verbose_level="NO_OUTPUT",
            manual_control=(policy != "native"),
        )
    except TypeError:
        # A TypeError means this script drifted from the env constructor
        # signature — that is a bug to fix, not a scenario to skip.
        raise
    except Exception as exc:
        print(f"  [SKIP] init failed: {exc}")
        print()
        return None

    t_init = time.perf_counter() - t0
    n_ix = len(env.intersections)
    print(f"  Init           : {t_init:.1f} s  (GPU kernel compilation)")
    print(f"  Intersections  : {n_ix}")
    print(f"  Policy         : {_policy_label(policy, explore, learn, fixed_action)}")
    print(f"  Max steps/ep   : {max_steps}  ({max_steps * delta:,} sim-s)")
    print()

    # GPU warm-up: let the GPU reach steady-state throughput before timing.
    agent = _make_agent(env, policy, fixed_action, seed)
    print(f"  GPU warm-up: {warmup} steps ...", end="  ", flush=True)
    info = env.reset()
    t_wu = time.perf_counter()
    for _ in range(warmup):
        action = agent.act(info, explore=explore)
        reward, terminated, truncated, info = env.step(action)
        if learn:
            agent.observe(info, reward, bool(terminated), bool(truncated))
        if terminated or truncated:
            info = env.reset()
    print(f"done  ({time.perf_counter() - t_wu:.1f} s)")
    print()

    all_step: list[float] = []
    all_sim:  list[float] = []
    all_rewards: list[float] = []
    all_mets: list[dict]  = []

    for ep in range(1, n_eps + 1):
        step_ms, sim_ms, rewards, mets = _run_episode(env, agent, max_steps, learn, explore)
        all_step.extend(step_ms)
        all_sim.extend(sim_ms)
        all_rewards.extend(rewards)
        all_mets.append(mets)
        _ep_report(ep, n_eps, step_ms, sim_ms, rewards, mets, delta)

    env.close()
    return _summarise("MOSS", all_step, all_sim, all_rewards, all_mets, delta)


# Comparison table
def _compare(sumo: dict | None, moss: dict | None) -> None:
    _sep()
    print("  COMPARISON")
    _sep()

    col = 14

    def _row(
        label: str,
        sv,
        mv,
        fmt: str = ".1f",
        suffix: str = "",
        *,
        higher_better: bool = False,
        show_speedup: bool = True,
    ) -> None:
        if sv is None and mv is None:
            return
        s_str = f"{sv:{fmt}}{suffix}" if sv is not None else "N/A"
        m_str = f"{mv:{fmt}}{suffix}" if mv is not None else "N/A"
        if show_speedup and sv is not None and mv is not None and sv > 0 and mv > 0:
            spd = mv / sv if higher_better else sv / mv
            spd_str = f"{spd:.2f}x"
        else:
            spd_str = "-"
        print(f"  {label:<32}  {s_str:>{col}}  {m_str:>{col}}  {spd_str:>8}")

    def _traffic_row(label: str, key: str, fmt: str = ".1f", higher_better: bool = False) -> None:
        sv = sumo[key] if sumo else None
        mv = moss[key] if moss else None
        if sv is None and mv is None:
            return
        s_str = f"{sv:{fmt}}" if sv is not None else "N/A"
        m_str = f"{mv:{fmt}}" if mv is not None else "N/A"
        if sv is not None and mv is not None and abs(sv) > 1e-9:
            diff = (mv - sv) / abs(sv) * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "-"
        print(f"  {label:<32}  {s_str:>{col}}  {m_str:>{col}}  {diff_str:>8}")

    # Header
    print(f"  {'Metric':<32}  {'SUMO':>{col}}  {'MOSS':>{col}}  {'Speedup':>8}")
    _hsep()

    # Speed
    _row("ms / step",           sumo["ms"]      if sumo else None,
                                moss["ms"]      if moss else None,
                                ".1f", " ms")
    _row("steps / second",      sumo["sps"]     if sumo else None,
                                moss["sps"]     if moss else None,
                                ".2f", higher_better=True)
    _row("sim-s / real-s",      sumo["sim_sps"] if sumo else None,
                                moss["sim_sps"] if moss else None,
                                ".0f", higher_better=True)
    _row("reward / step",       sumo["reward_step"] if sumo else None,
                                moss["reward_step"] if moss else None,
                                ".3f", show_speedup=False)
    _row("total return",        sumo["return"] if sumo else None,
                                moss["return"] if moss else None,
                                ".2f", show_speedup=False)

    # Overhead breakdown
    print()
    print(f"  {'Timing breakdown':<32}  {'SUMO':>{col}}  {'MOSS':>{col}}")
    _hsep()
    def _ov_row(label, sk, mk, fmt=".1f", suffix=""):
        sv = sumo[sk] if sumo else None
        mv = moss[mk] if moss else None
        s_str = f"{sv:{fmt}}{suffix}" if sv is not None else "N/A"
        m_str = f"{mv:{fmt}}{suffix}" if mv is not None else "N/A"
        print(f"  {label:<32}  {s_str:>{col}}  {m_str:>{col}}")
    _ov_row("simulate (engine)",  "sim_ms", "sim_ms", ".1f", " ms")
    _ov_row("Python overhead",    "py_ms",  "py_ms",  ".1f", " ms")
    if sumo and moss:
        s_pct = sumo["py_ms"] / sumo["ms"] * 100 if sumo["ms"] > 0 else 0
        m_pct = moss["py_ms"] / moss["ms"] * 100 if moss["ms"] > 0 else 0
        print(f"  {'Python overhead %':<32}  {s_pct:>{col}.1f}%  {m_pct:>{col}.1f}%")

    # Traffic parity
    print()
    print(f"  {'Traffic parity (last episode)':<32}  {'SUMO':>{col}}  {'MOSS':>{col}}  {'MOSS delta':>10}")
    _hsep()
    _traffic_row("avg travel time (s)",       "tt",       ".1f")
    _traffic_row("vehicles completed",        "compl",    ".0f", higher_better=True)
    _traffic_row("avg pressure (veh/ix)",     "pressure", ".3f")

    # Verdict
    print()
    if sumo and moss:
        speedup = sumo["ms"] / moss["ms"]
        _sep("=")
        if speedup >= 1.0:
            print(f"  MOSS is {speedup:.1f}x faster than SUMO on this hardware.")
        else:
            print(f"  MOSS is {1.0 / speedup:.1f}x slower than SUMO on this run.")
        print(f"  MOSS throughput : {moss['sim_sps']:,.0f} simulated seconds / real second")
        print(f"  SUMO throughput : {sumo['sim_sps']:,.0f} simulated seconds / real second")
        _sep("=")
    print()


# Main
def main() -> None:
    args = _parse()
    scenario = _scenario_paths(args.scenario)
    _apply_path_overrides(scenario, args)

    _sep("=")
    print(f"  SUMO vs MOSS Performance Benchmark  -  {scenario['label']}".center(_W))
    _sep("=")
    print()
    print(f"  GPU  : {_gpu_info()}")
    print(f"  CPU  : {_cpu_info()}")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print()
    sim_total = args.episodes * args.steps * args.delta
    learn = not args.no_learn
    explore = not args.eval
    if args.policy in {"fixed", "native"}:
        learn = False
        explore = False
    print(f"  Config  : {args.episodes} episodes  x  {args.steps} steps/ep"
          f"  x  {args.delta} s/step  =  {sim_total:,} sim-s per backend")
    print(f"  Scenario: {args.scenario}")
    print(f"  Policy  : {_policy_label(args.policy, explore, learn, args.fixed_action)}")
    if args.policy == "idqn":
        print(f"  Agent   : seed={args.seed}")
    print()

    t_wall = time.perf_counter()
    sumo = bench_sumo(
        args.episodes,
        args.steps,
        args.delta,
        scenario,
        seed=args.seed,
        learn=learn,
        explore=explore,
        policy=args.policy,
        fixed_action=args.fixed_action,
    )
    moss = bench_moss(
        args.episodes,
        args.steps,
        args.delta,
        args.warmup,
        args.device,
        scenario,
        seed=args.seed,
        learn=learn,
        explore=explore,
        policy=args.policy,
        fixed_action=args.fixed_action,
    )

    _compare(sumo, moss)

    total = time.perf_counter() - t_wall
    print(f"  Total benchmark wall time : {total:.1f} s  ({total/60:.1f} min)")
    print()


if __name__ == "__main__":
    main()
