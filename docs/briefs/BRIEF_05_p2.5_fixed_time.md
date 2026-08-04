# BRIEF #5 — P2.5: fixed-time controller (ladder Tier 1)

**Mode:** Claude Code, in the repo, on a task branch.
**Branch:** `task/p2.5-fixed-time`
**Supersedes:** the "Brief #2c" placeholder in PROJECT_PLAN §6 (never written) and the P2.5 line there.
**Issued:** 2026-08-04 by the Master chat.

---

## 1. Why this task exists, and why it is not a CLI flag

No fixed-time controller exists in this repo (verified 2026-07-25, zero matches for
`fixed.time|FixedTime`; re-verified 2026-08-04). It is **Tier 1 of the C1 dataset ladder**, and the
first item on the paper's scientific path after a long run of infrastructure work.

It is also not the weakest rung — frame it as the **most deployment-realistic data source**. DataLight
builds an entire case study (COD, "cyclical offline data") on FixedTime precisely because *"in
real-world TSC scenarios, FixedTime stands out as the most commonly employed method"*, and their
offline models trained on cyclical logs beat BC/BCQ/CQL by 8.5–26.7%. Cyclical logs are what an actual
city can hand you. That framing is the point of Tier 1, not ladder completeness.

---

## 2. The finding that reshapes this task: the scenarios already ship a real fixed-time plan

All four hangzhou variants ship `signal_plan_template.txt`, which **no Python in this repo
references** (grepped 2026-08-04). Measured content, identical across the variants checked:

```
header line          : intersection_1_1
3600 rows            : one phase index per simulated second
distinct phases      : 0..8    (0 = clearance, 1..8 = greens)
run-length structure : (green, 30 s) then (0, 5 s), repeating
cycle                : 8 greens x (30 + 5) = 280 s
```

**So the canonical cycle length and phase split do not need inventing — the scenario supplies them:
equal 30 s greens, 5 s clearance.** Using them makes Tier 1 genuinely the plan a city would run,
rather than something we chose.

Two caveats you must handle rather than assume away:

1. **Only hangzhou ships one.** cologne1, cologne3 and grid4x4 do not. They need a principled default
   (see §4.3).
2. **The plan's phase indices are FILE phase indices**, while acyclic actions index the *green list*
   (phases whose file duration exceeds `TRANSITION_PHASE_MAX_DURATION = 5`). The mapping looks like
   `green action a ↔ file phase a+1`, **but verify it from `IntersectionInfo` rather than assuming
   it** — a silent off-by-one here would mis-label every Tier 1 trajectory, and the resulting corpus
   would look perfectly valid.

---

## 3. The hard part: the shipped plan does not tile the decision grid

`delta_time = 10` in every config in this repo (verified: all 6 files under `experiments/configs/`,
and `collect.py`'s default). The shipped plan is **30 s green + 5 s clearance = 35 s per phase**, and
**35 mod 10 = 5**. A policy that acts every 10 s therefore *cannot* reproduce the shipped plan exactly.

What the platform actually renders under `acyclic` (from `docs/phase-control.md` and
`envs/phase_control.py`): choosing a *different* green plays a forced clearance first — CityFlow
all-red 5 s, SUMO yellow 3 s + all-red 2 s — then the target green fills the rest of `delta_time`.
Choosing the *same* green just continues it. So holding a phase for `k` decision steps and then
switching yields, per phase:

```
5 s clearance + (10k - 5) s green      cycle = 8 x 10k seconds
k = 3  ->  25 s green + 5 s clearance,  240 s cycle
k = 4  ->  35 s green + 5 s clearance,  320 s cycle
shipped ->  30 s green + 5 s clearance, 280 s cycle   (unreachable on a 10 s grid)
```

**Do not resolve this by changing `delta_time`.** A tier collected at a different decision interval
would break the ladder: C1 compares datasets, the DT's context length K counts decision steps, and a
tier with a different step duration silently changes the wall-clock horizon K represents. `delta_time`
is constant across the ladder. (A `delta_time = 5` faithful-reproduction run is legitimate as a
*sensitivity* measurement, and §4.4 asks for it, but it does not enter the corpus.)

**Choose k by measurement, not by taste.** k = 3 and k = 4 bracket the shipped plan and are equidistant
on both green time (±5 s) and cycle length (±40 s), so no arithmetic argument settles it. Run both,
compare against the shipped plan's own behaviour, and pick the closer one on `average_travel_time`.
Report both numbers and the deviation you accepted — a reader must see that the tier is an
*approximation of* the shipped plan, and by how much.

---

## 4. Scope — four deliverables

### 4.1 `offline/policies/fixed_time.py` (NEW)
A fixed-time controller registering into `collect.py`'s `POLICIES` registry as `"fixedtime"`.

Interface, mirrored from the existing entries — `POLICIES[name](env, args, rng)` returns a callable
taking `info` and returning `np.ndarray` of `int64`, one action per intersection, ordered by
`[ix.id for ix in env.intersections]`:

```python
POLICIES: dict[str, PolicyFactory] = {
    "maxpressure": _make_maxpressure, "random": _make_random,
    "mappo": _make_mappo, "mappo_eps": _make_mappo_eps,
}
```

Requirements:
- **Support `acyclic` at minimum.** `env.control_mode` decides the vocabulary; mirror
  `algorithms/max_pressure.py`, which switches on it. If you support only `acyclic`, **raise a clear
  error** on the others rather than silently emitting binary actions — a fixed-time controller that
  quietly does the wrong thing under `cyclic` is exactly the silent-semantic-bug class this repo
  exists to prevent.
- **Respect `avail_actions`.** The env raises on an illegal action. Under `acyclic` every green is
  always available, but do not rely on that — read the mask, and if the scheduled phase is not
  available, say what you do about it in a docstring.
- **Deterministic and stateless w.r.t. wall clock:** the phase must be a pure function of the step
  index (or `info["step"]` / `sim_time`), never of an internal counter that drifts if a caller skips
  a step. Two runs with the same seed and draw must produce byte-identical trajectories — the corpus
  determinism guarantee depends on it.
- The offset/phase at t=0 is a real parameter. Say what you chose and why.

### 4.2 Read the shipped plan
Parse `signal_plan_template.txt` when present, derive greens and durations from it, and map file
phases to green-action indices **verified against `IntersectionInfo`**, not assumed. If the file is
absent, fall back to §4.3.

### 4.3 The no-plan default
cologne1 / cologne3 / grid4x4 ship no plan. Default: **equal split across the intersection's green
phases**, with the same `k`-step hold as hangzhou, clearance handled by the platform. State the cycle
length this produces per scenario. This is a declared choice; put it in the docstring and the packet.

### 4.4 Measurements to report (not to decide on)
- k = 3 vs k = 4 on hangzhou bc-tyc, CityFlow, `average_travel_time` **over all vehicles that
  entered** plus completion and entered counts (`PREREGISTRATION.md` §3.1 — do not use a
  completed-trips-only metric).
- The same scenario under the **shipped plan's own timing** at `delta_time = 5`, as the faithfulness
  reference. This one is a sensitivity measurement and does **not** enter the corpus.
- Compare against the P0.2 anchors (`docs/p0_baseline_numbers.md`): MaxPressure 160.56, Random 307.53
  on cf_hz1x1. **Sanity expectation: fixed-time should land between MaxPressure and Random.** If it
  beats MaxPressure or loses to Random, stop and report — that is a bug signal, not a finding.

---

## 5. Scope fence

- **No collection campaign.** This task delivers the controller and its numbers, not a corpus. P2.2
  runs the campaign.
- **No changes to `envs/`, `algorithms/`, or any frozen file.** No authorisation is granted here. If
  `phase_control.py` seems to need a change, that is a Return Packet open question.
- **Do not change `delta_time` for anything that would enter the corpus** (§3).
- **Do not implement Webster's formula or any demand-responsive timing.** Fixed-time means fixed; a
  demand-adaptive cycle is a different policy and would contaminate the ladder's Tier 1 semantics.
- **No new dependencies.**

---

## 6. Tests, including the load-bearing one

- **Load-bearing:** the phase-index mapping. Assert that the action the controller emits for a given
  scheduled *file* phase actually results in that phase being active in the env, by stepping a real
  env and reading back `info["intersections"][id]["current_phase"]`. Prove it can fail by mutating the
  mapping by one and showing the test catches it. **This is the test that stops a silently mis-labelled
  Tier 1 corpus**, which would look valid and be wrong.
- Determinism: two runs, same seed and draw, byte-identical action sequences.
- Cycle structure: over N steps the emitted sequence has the expected period and per-phase hold, and
  every phase appears (a controller stuck on one phase must fail).
- Parser: a malformed/absent `signal_plan_template.txt` falls back to §4.3 rather than crashing or
  silently emitting phase 0.
- Standard hygiene: no reasonless skips, no `pytest.raises` without `match=`, no weakened assertions.

---

## 7. Definition of Done

- [ ] Plan file first (`docs/plans/p2.5.md`), approved before code
- [ ] `offline/policies/fixed_time.py` + registry entry; `--policy fixedtime` works end to end
- [ ] Load-bearing mapping test written, shown to fail against a mutation, then passing
- [ ] k = 3 vs k = 4 measured and a recommendation made (the Master chat rules on which ships)
- [ ] Sanity check against P0.2 anchors reported explicitly
- [ ] Full `.venv/bin/pytest -q`, real tail, count reported against the current 270
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P2.5.md`

## 8. Return Packet — task-specific questions

1. The verified file-phase ↔ green-action mapping, with the evidence you used. Did it match `a+1`?
2. k = 3 vs k = 4 numbers, with entered/completed counts, and which you recommend and why.
3. Where fixed-time lands relative to MaxPressure (160.56) and Random (307.53) on cf_hz1x1.
4. What you chose for the t=0 offset and the no-plan default, and what else you considered.
5. Anything in §3's reasoning you think is wrong. The last three tasks each corrected the
   coordinator and each correction stood.

---

## 9. Rulings on the plan-mode questions (Master chat, 2026-08-04)

### Ruling 1 — the `delta_time = 5` faithfulness reference in §4.4 is impossible. My error. Direct replay approved, **with a validation gate.**

Confirmed at `envs/phase_control.py:480`: `if len(self._green_phases) > 1 and transition_duration >=
self._delta_time: raise`. CityFlow's clearance is 5 s, so `5 >= 5` raises. No phase-control mode can
command a full decision step of clearance, so §4.4's reference run cannot exist as written. Third
brief defect found by an implementer in three tasks; the finding is correct and the brief is wrong.

**Direct 1 s replay of `signal_plan_template.txt` on a CityFlow engine is approved as the ground
truth** for choosing k.

**But it must be validated before its number is used**, because the replay harness and the env compute
`average_travel_time` by *different code paths*, and k would then be chosen by comparing across
measurement pipelines. Cheapest sufficient validation, and it is required:

> Pick a degenerate plan that both routes can express — e.g. **hold green 1 for the entire episode**.
> Run it through the env (acyclic, `delta_time=10`, action 0 at every step) and through the replay
> harness, on the same scenario and demand draw. **The two `average_travel_time` values must agree.**
> If they do not, the replay number is not comparable to the k=3 / k=4 numbers and must not be used to
> choose k — report the discrepancy instead and stop.

This is the double-computation rule applied to a measurement path rather than to a quantity.

### Ruling 2 — record `k` in the manifest. Do **not** defer it to P2.2.

The blast radius you were routing around does not exist, and 20 seconds of checking would have shown
it:

- **`epsilon` is an exact precedent** — `offline/collect.py:563`,
  `"epsilon": float(args.epsilon) if args.policy == "mappo_eps" else None` — a policy-specific field
  recorded as `None` when inapplicable. Follow it exactly.
- **The manifest test is not exhaustive.** `tests/test_trajectory_logger.py:609-624` asserts named
  keys (`format_version`, `lane_count`, `lane_ids_sha256`, `run_metadata["scenario"]`, `git_hash`) and
  never asserts a complete key set. Adding a field breaks nothing and touches no frozen file.

Why this is not optional: without `k` in the manifest, **two Tier 1 runs with different `k` are
indistinguishable from the data**, and the tier's cycle timing exists only in whatever session
produced it. That is precisely the N5 defect class the project just spent a task repairing. Also
record which schedule source was used (shipped plan vs equal-split fallback) and, when the shipped
plan is used, its **sha256** — the corpus should be able to prove which plan it came from.

*Caution worth naming: "I'll flag it rather than risk it" is the right instinct, but the rule here is
check-then-decide. An assumed blast radius is an assumption like any other.*

### Ruling 3 — acyclic only, hard error elsewhere. Confirmed.

Same invariant as `delta_time` in §3: the ladder is collected under **one** control mode, C1 compares
datasets against each other, and a tier collected under a different action vocabulary is not
comparable to the others. The error message must state *why* (ladder comparability), not merely that
the mode is unsupported — an error that explains itself is what stops the next person from "fixing" it
by adding a silent fallback.

Minor packet note, not blocking: PROJECT_PLAN §3 claims the platform already offers "fixed-time via
phase-control modes". Check briefly whether that referred to a `cyclic`-mode schedule, and say so in
the packet — if it is stale, I will correct §3.

### Not asked, but approved as proposed
Stateless schedule as a pure function of `info["step"]`; greens derived from `IntersectionInfo` with
the `a+1` mapping **verified live rather than assumed** (good); cycle-index 0 at step 0 so no spurious
initial clearance; mask fallback documented. Proceed on all of these.
