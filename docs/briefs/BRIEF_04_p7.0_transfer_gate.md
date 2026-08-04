# BRIEF #4 — P7.0: the C3 early gate, with vehicle-parameter parity as a precondition

**Mode:** Claude Code, in the repo, on a task branch.
**Branch:** `task/p7.0-transfer-gate`
**Supersedes:** the P7.0 line in PROJECT_PLAN §6 and the (superseded) 2026-07-31 landmine row.
**Issued:** 2026-08-04 by the Master chat.

> **Read §2 before planning.** The prerequisite this task was scoped around was measured on
> 2026-08-04 and found to be a 0.6% effect. The real one is different, larger, and is not fixed by
> what the old plan told you to do. Working from the old description will produce a confounded gate.

---

## 1. What P7.0 is, and what it decides

P7.0 is the **~1-day kill-switch for claim C3**. It logs MaxPressure trajectories on one paired
scenario in **both** backends and compares state-feature distributions and MaxPressure-normalised
returns. A pathological shift descopes C3 to a characterised-limitations study before we invest in the
full transfer curve (P7.1–P7.4).

It is a gate, not a result. Its output is a go/no-go plus a measured description of the gap. It must
therefore be **cheap and honest**, not impressive.

---

## 2. The precondition, and why the old one was wrong

`PREREGISTRATION.md` §9 registers a prerequisite: bind hangzhou's vType before running the gate,
because otherwise SUMO runs at 55.55 m/s against CityFlow's 11.11 and the gate measures a 5×
speed-limit artifact.

**That is refuted.** Measured 2026-08-04, SUMO 1.27.1, full record in
`docs/notes/P7.0_vtype_investigation.md`:

| hangzhou bc-tyc, 3600 s, identical net + TLS | arrived | mean TT (s) |
|---|---|---|
| as shipped (`DEFAULT_VEHTYPE`) | 1578 | 276.45 |
| bind the existing `pkw` vType | 1564 | 278.17 **(+0.6%)** |
| bind **+ `tau=2.0`** (CityFlow's `headwayTime`) | 1201 | 415.04 **(+49%)** |

The 55.55 ceiling can never bind: every hangzhou lane is capped at ≤ 11.11 m/s and `speedFactor` is
exactly 1.0. The dominant confound is **headway** — SUMO defaults `tau` to 1.0 while CityFlow
specifies `headwayTime` 2.0, and **the shipped `pkw` vType does not set `tau` either**. Binding the
vType and declaring the prerequisite satisfied would have left a ~50% travel-time confound inside the
gate.

⚠️ Those magnitudes are `tripinfo` means over **arrived** vehicles and the arrival counts differ by
23%, so by `PREREGISTRATION.md` §3.1's own >5% rule they establish **direction, not size**. Part of
your job is to redo this properly (§4).

---

## 3. The parity contract (RULING, 2026-08-04 — not negotiable in this task)

**Matched**, because both engines expose them: `tau`/`headwayTime`, `accel`, `decel`, `maxSpeed`,
`length`, `minGap`.
**Direction: configure the SUMO side to match CityFlow, on both hangzhou and cologne.** Not because
CityFlow is authoritative — on cologne it demonstrably is not — but because **CityFlow is the training
domain and the substrate of the whole C1/C2 corpus, so it must not move.**
**`speedFactor` → 1.0** (a sampling distribution over an already-matched parameter; CityFlow has
exactly one vehicle parameter set and cannot represent any spread).
**`sigma` stays native and is declared unmatchable** — it is a *model* parameter, and zeroing it would
replace SUMO's car-following model rather than align it. **Match parameters, never models.**
**Declared unmatchable, in both directions:** CityFlow `usualPosAcc` / `usualNegAcc` (no SUMO
counterpart); SUMO `sigma` and `speedFactor` distribution (no CityFlow counterpart).

Measured parity targets (read from the scenario files 2026-08-04, re-verify before use):

| | length | minGap | maxSpeed | accel | decel | headway/tau |
|---|---|---|---|---|---|---|
| hangzhou ×4 (CityFlow `flow.json`) | 5.0 | 2.5 | 11.11 | 2.0 | 4.5 | **2.0** |
| cologne1 / cologne3 (CityFlow) | 4.3 | 1.5 | 13.39 | 2.6 | 4.5 | **1.5** |

**Cologne's declared cost:** matching there caps SUMO vehicles at 13.39 m/s although cologne's own
lanes allow 13.89 and 19.44, and switches off `speedFactor`. Our parity-run cologne numbers are
therefore **not** comparable with published SUMO cologne results — the shipped-config sensitivity run
is what restores that comparison. Note CityFlow's 13.39 cap sits *below its own lane limits*, so it
binds on every road; that is a conversion artifact, not a modelling choice, and the paper should say so.

---

## 4. Scope — four deliverables, in order

### 4.1 Parity as a committed artifact (RULING condition 3)
A committed file carrying the matched values **with a comment naming the source of each** (which
`flow.json`, which key). Not a note, not a docstring, not a session scratchpad — P7.3 must be able to
consume it without re-deriving it. This is the N5 defect class; do not repeat it.

Suggested home: `offline/parity.py` or a JSON under `offline/` — your call, argue it in the plan. It
must be importable by `render_sumo` and readable by a human.

### 4.2 Parity rendering in `offline/flow_randomizer.py::render_sumo`
That function currently copies the template `<vType>` verbatim and mirrors its binding — correct
mirroring, and the right single place for an explicit parity type. **`offline/` is ours, not frozen.**
No SUMO corpus exists yet, so changing it now costs nothing.

Requirements: emit a parity `<vType>` from §4.1 and bind it on **every** `<vehicle>`; keep the existing
mirroring behaviour available for the sensitivity runs (a flag, defaulting to parity **off** so no
existing caller silently changes behaviour — argue the default in your plan if you disagree).

### 4.3 Redo the A/B properly
Re-run the §2 comparison using `average_travel_time` **over all vehicles that entered**, not `tripinfo`
means over arrivals, and co-report completion and entered counts per `PREREGISTRATION.md` §3.1. This
replaces my direction-only figures with quotable ones. If the entered counts differ by >5% between
variants, report the comparison invalid rather than quoting a magnitude — that rule applies to us.

### 4.4 The gate itself
Log MaxPressure in **both** backends on hangzhou bc-tyc under parity: per-feature state distribution
comparison (KS statistic and/or overlap coefficient per feature) plus MaxPressure-normalised returns.
Report the numbers and a go/no-go recommendation. **You do not decide the go/no-go** — you report; the
Master chat rules.

---

## 5. Scope fence — what NOT to build

- **No transfer training.** P7.0 is a measurement gate; no DT, no fine-tuning, no transfer curve.
- **Do not modify CityFlow-side scenario files or `flow.json`.** The training domain does not move.
  If you believe a CityFlow file is wrong, that is a Return Packet open question, not an edit.
- **Do not touch frozen files.** No authorisation is granted by this brief. `offline/` and `tests/`
  are yours; `envs/`, `metrics/`, `scenarios/` are not.
- **No new dependencies.** `traci` is already installed and used by the repo; SUMO 1.27.1 is on PATH.
- **Do not "fix" cologne's 13.39 cap** in either engine. It is recorded as a conversion artifact and
  handled by the parity contract; changing it is a separate decision.
- **Do not run the full corpus.** One scenario, both backends, enough episodes for a distribution
  comparison.

---

## 6. Tests, including the load-bearing one

- **Load-bearing:** a test that renders a parity `.rou.xml` and asserts **every** `<vehicle>` carries
  the parity `type`, and that the emitted `<vType>` attributes equal the §4.1 artifact exactly. Prove
  it can fail: mutate one attribute and show the test catches it. The whole parity ruling rests on this
  binding actually happening on every vehicle — hangzhou shipped 2021 vehicles with a correct vType
  defined and bound to **none** of them, which is exactly the failure this test exists to catch.
- Round-trip: parity rendering off ⇒ byte-identical to current behaviour (protects the sensitivity run
  and every existing caller).
- The parity artifact's values match the scenario `flow.json` files they claim to come from — read
  both, compare, so the committed table cannot silently drift from its source.
- Standard hygiene: no reasonless skips, no `pytest.raises` without `match=`, no weakened assertions.

---

## 7. Definition of Done

- [ ] Plan file first (`docs/plans/p7.0.md`), approved before code
- [ ] Parity artifact committed, each value's source named
- [ ] `render_sumo` parity path + flag; existing behaviour unchanged by default
- [ ] Load-bearing test written, shown to fail against a mutation, then passing
- [ ] §4.3 redone with `average_travel_time` over entered vehicles; entered/completed co-reported
- [ ] Gate numbers reported: per-feature KS/overlap + MaxPressure-normalised returns, both backends
- [ ] Full `.venv/bin/pytest -q` run, real tail pasted, count reported against the current 270
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P7.0.md`

## 8. Return Packet — task-specific questions

1. Did any parity target in §3 disagree with the scenario file when you re-read it? Quote both.
2. Your §4.3 numbers, with entered and completed counts. State whether the >5% rule invalidates any
   comparison you report.
3. The gate numbers, with **no** go/no-go verdict — that ruling is mine.
4. Anything in the parity contract you believe is wrong. You are expected to push back; the last two
   tasks both corrected the coordinator, and both corrections were right.
