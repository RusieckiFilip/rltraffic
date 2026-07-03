# Phase control

`envs/phase_control.py` defines *how agent actions map to traffic-light
phases*. The env is constructed with one `PhaseControl` strategy class
(`phase_control_cls`, or `control_mode` in experiment configs) and creates
one instance per intersection.

Each decision step, the control turns the chosen action into a
`PhasePlan` — an ordered tuple of `PhaseSegment(phase, duration, kind)`
whose durations sum to exactly `delta_time`. Segment kinds:

- `"phase"` — a real controllable phase index;
- `"yellow"` / `"all_red"` — *synthetic transition* segments rendered by the
  backend (SUMO shows real yellow; CityFlow/MOSS use an injected all-red
  phase).

The base env validates every plan (positive durations, in-range phases,
total == `delta_time`) and exposes the currently legal actions per
intersection as `info["intersections"][id]["avail_actions"]`. Choosing an
action outside that list raises.

## Modes

### `acyclic` — `AcyclicPhases` (default)

The action directly selects a **green phase**; any green can follow any
other. Greens are the phases whose configured file duration exceeds 5 s
(`TRANSITION_PHASE_MAX_DURATION`); the scenario's own short yellow/clearance
phases are never selectable.

Switching greens always plays a forced clearance first:

- SUMO: yellow 3 s → all-red 2 s → target green for the rest of `delta_time`;
- CityFlow / MOSS: all-red 5 s → target green.

Because the clearance already enforces safe timing, min/max phase bounds are
ignored — every green is available at every tick. Action count = number of
green phases. Requires `delta_time > 5` (the transition must fit strictly
inside the decision window when there is more than one green).

### `acyclic_bounded` — `AcyclicBoundedPhases`

Same transition behaviour as `acyclic`, but the configured min/max phase
duration bounds gate availability: the current green must be held for at
least its minimum duration, and once its maximum is reached the agent is
forced to switch (the current action disappears from `avail_actions`).

### `cyclic` — `CyclicPhases`

Binary action: `0` = keep the current phase, `1` = advance to the next phase
in the scenario's fixed cycle (modulo phase count; transitions come from the
scenario file itself, no synthetic segments). Min/max bounds restrict the
choice: below min only `keep` is available, at max only `switch`.

### `resco_cyclic` — `RescoCyclicPhases`

RESCO-compatible cyclic control. Also binary (`keep` / `next green`), but:

- greens are detected from phase metadata (has active road links, duration
  > 5 s, state string contains no `y`);
- on `switch`, the scenario's own transition phases between the current and
  next green (typically yellow + red) are played with their *configured file
  durations*, then the next green fills the rest of `delta_time`;
- a `switch` request before `min_green + transition` time has elapsed in the
  current green is silently treated as `keep` (RESCO semantics: both actions
  are always reported available);
- requires at least two green phases and a fixed `delta_time` longer than
  the transition.

## Phase duration bounds

`env.set_phase_durations(bounds)` accepts an
`(n_intersections, max_phases, 2)` int array (`[..., 0]` = min,
`[..., 1]` = max seconds). Bounds are honoured by `acyclic_bounded`,
`cyclic` and `resco_cyclic` (as min-green). When a mode needs bounds and
none were provided, the env builds defaults from the scenario metadata:
green phases get `[5, 60]` s, transition phases are pinned to 5 s
(`GREEN_MIN_DURATION`/`GREEN_MAX_DURATION`/`YELLOW_DURATION` in
`envs/base_traffic_env.py`).

## Execution across intersections

`BaseTrafficEnv._execute_phase_plans` runs all intersections' plans in
lockstep: it repeatedly advances the simulator by the shortest remaining
segment duration and applies whichever segments start at that boundary, so
intersections with different plan structures stay synchronized within the
same `delta_time` window.
