# BRIEF_35 — P7.2a: SUMO draws under the parity contract — bound route rendering and teleport-free `.sumocfg` generation for materialised draws

**Task id:** `P7.2a` · **Branch:** `task/p7.2a-sumo-draws`, cut from `main` **after** `git fetch` (the commit that carries this brief; `git log -1 --format=%h origin/main` — a header cannot know its own sha) · **Issued:** 2026-09-12
**Mode:** Claude Code, plan mode first, worktree `/home/filip/rltraffic-p53b` (`git -C /home/filip/rltraffic-p53b checkout -b task/p7.2a-sumo-draws main` — the worktree is on the merged `task/p7.1-alignment`; creating a new branch from `main` there works, checking out `main` does not)
**Registered as:** the first executable half of `PROJECT_PLAN` §6 **P7.2** (*RTG calibration protocol for cross-domain prompting*). The protocol's target-domain probe (P7.2b, next brief) must run MaxPressure on the **same demand draws** P4.3's in-domain probe used (`docs/data/p4_3_probe.json`: draws **201–300**), on SUMO, under the parity vType and the teleport-free regime. **No such SUMO demand exists on disk**: `scenarios/draws/cityflow1x1/*/routes.rou.xml` mirrors the shipped file's unbound `pkw` (`offline/flow_randomizer.py:492`), no `.sumocfg` is generated (`offline/materialise_draws.py:113-121`), and `DEFERRED` 75 (a)+(b) owns exactly that gap. This brief closes (a) and the `.sumocfg` half of (b).
**Pre-registration that binds:** **A15(c)** — every SUMO measurement from 2026-09-12 on runs a configuration carrying `<time-to-teleport value="-1"/>`, and *every `.sumocfg` generated for drawn demand carries the same element*; **A14(E)** — demand per vehicle exact, vehicle parameters by the parity contract (2026-08-04), the shipped or converter `vType` never used; **A15(g)** — hz1x1 bc-tyc is the ADMITTED C3 pair. **A16** (feature freeze) is not touched by this task.
**Compute:** no training, no campaign under `output/`. Rendering is pure computation (≈ 206 draws, seconds). Tests run **at most three SUMO episodes** (≈ 10–12 s each bare, `n = 5` per arm in `docs/data/p7_1_metric_freeze.json`) and **one CityFlow episode** (≈ 1 s) — SUMO-gated tests self-skip when SUMO is absent, so **CI's skip ceiling will move** (`tests/test_ci_gate.py::CEILING_CHAIN`, registered route: classify `junit.xml`, commit the observed value — the coordinator does this at merge).
**Contracts:** `docs/CONTRACTS.md` v1.1 — C1, C2 bind for the runtime checks. `envs/`, `utils/`, `metrics/`, `scripts/`, `.claude/` are FROZEN. Everything here lives in `offline/` and `tests/`.
**Cost:** ~1 day including the pre-flight and the two verification runs. `DEFERRED` 75 estimated (a)–(d) at ~1 day; this brief is (a) plus half of (b), scoped to one scenario.

> **Read order:** this brief → `offline/materialise_draws.py` module docstring and `_existing_conflict` / `_commit` (`:591-692`) → `offline/parity.py` module docstring, `render_parity_rou_text` (`:260`), `_verify_rendered_rou` (`:305`), `render_parity_sumocfg_text` (`:332`), `write_parity_scenario` (`:405`) → `offline/flow_randomizer.py::render_sumo` (`:492`) → `docs/notes/DEFERRED.md` rows 55 and 75 → `PREREGISTRATION.md` §12 rows A14(E), A15(c), A15(g) → `docs/data/p4_3_probe.json` (`draw_ids`, `engine_seed`, `env_settings`, `episodes[0]`) → `tests/test_materialise_draws.py` and `tests/test_parity_vtype.py` (the shapes to extend). Where this brief disagrees with the repo, **the repo wins and you flag it** (`CLAUDE.md` §2).

---

## 0. What the coordinator verified before writing this (2026-09-12, by running commands, not by reading docstrings)

1. **`scenarios/draws/cityflow1x1/` holds draws 0–5 and 1000–1099 — 106 directories, four files each** (`flow.json`, `cityflow.json`, `routes.rou.xml`, `provenance.json`). **Draws 201–300 — P4.3's probe band — are NOT on disk.** They were materialised in a since-retired worktree (`DEFERRED` 55's mechanism); `docs/data/p4_3_probe.json` carries no per-draw flow hash and no draw path, only per-draw returns. Regeneration is a pure function of `(source, base_seed = 1000, draw_id)` (`offline/materialise_draws.py` docstring, *DRAW IDENTITY*), and §5 below turns P4.3's recorded return into the regeneration gate.
2. **The rendered `routes.rou.xml` of every existing draw declares exactly one `<vType id="pkw">` and binds it to nobody** — draw 1000: 1,821 `<vehicle>` elements, 0 with a `type=` attribute. That is precisely the input `offline.parity.render_parity_rou_text` accepts (exactly one `<vType>`, no vehicle already typed, `:260-290`) and refuses otherwise. **The parity binding of a rendered draw is therefore that function applied to the rendered text — reuse, no new renderer.** `_verify_rendered_rou` (`:305`) proves vehicle ids, departures and routes survive element for element.
3. **`_existing_conflict` (`offline/materialise_draws.py:591`) compares FILES only** — `present = {p.name for p in target.iterdir() if p.is_file()}` — so **a subdirectory inside an existing draw directory is invisible to the no-op identity check**, while any new *file* placed directly in the draw directory makes re-materialisation refuse (`it holds unexpected file(s)`). This decides the layout in §3.1: the parity artifacts live in `draw_NNNN/parity/`, and the parent's four files are never touched.
4. **`_commit` (`:644`) moves a staged directory onto the target with `os.replace`; `force` moves the old directory aside and unlinks it after the new one is in place.** A wrong target path in that `os.replace` would replace a *draw* instead of a *parity subdirectory* — the draws 1000–1099 are what every merged held-out number since P4.6 resolves through (`DEFERRED` 55). **That is why §6 requires a pre-flight review even though nothing is written under `output/`.**
5. **`render_parity_sumocfg_text` (`offline/parity.py:332`) emits `begin 0`, `end 4000`, no `<processing>` block.** The committed teleport-free config (`scenarios/hangzhou_1x1_bc-tyc_18041610_1h_parity/hangzhou_1x1_bc-tyc_18041610_1h_parity_noteleport.sumocfg`, `18ed5cb`) was written by hand at P7.1 with `<processing><time-to-teleport value="-1"/></processing>`. `tests/test_parity_vtype.py::test_parity_rou_regenerates_byte_identically` and the three `test_parity_sumocfg_*` tests pin the committed P7.0 artifacts, so **the generator gains an optional parameter whose default reproduces today's bytes exactly**.
6. **The draw pipeline's SUMO rendering is demand-faithful by construction but has never been audited per draw.** `render_sumo` sorts by `startTime`, numbers vehicles `0..n-1`, writes `depart = startTime + <begin>` with two decimals, and copies the template's `<vType>` verbatim. `offline/conversion_audit.py` already has the CAP(E) primitives — `demand_from_route_file` (`:292`), `demand_from_cityflow_flow` (`:317`) — and P7.1 ran them on the nominal pair (2021/2021, order matches). §3.2 runs the same comparison on every draw and records it.
7. **SUMO 1.27.1 via TraCI exposes `simulation.getOption` and `vehicle.getTypeID`** (checked in this venv on 2026-09-12; `libsumo` does not import, `DEFERRED` 76). So the two contract facts — *the effective type is `cf_parity`* and *teleporting is disabled* — can be asserted **at runtime, from the running engine**, not from the file that requested them. `SumoEnv` keeps the connection as `self._sumo` (`envs/sumo_env.py:130`) and seeds SUMO through `--seed` from `reset(seed=…)` (`:182-183`).
8. **`utils/sumo_utils.iter_sumo_input_paths` resolves relative `net-file` / `route-files` against the `.sumocfg`'s own directory**, and `SumoEnv.__init__` calls `validate_sumo_inputs_exist` (`:134`). A generated config under `draw_NNNN/parity/` therefore references the shipped network by a relative path (`os.path.relpath`, as `write_parity_scenario` already does) and its own bound route file by bare name.
9. **P4.3's probe settings** (`docs/data/p4_3_probe.json`): `engine_seed 1000`; `env_settings` = `max_steps 360`, `delta_time 10`, `control_mode acyclic`, `local_reward_fn queue_length`, `global_reward_fn queue_length`, `global_reward_weight 0.0`, `state_features [lane_vehicle_count, lane_waiting, phase_onehot]`, `thread_num 1` — identical to `offline.transfer_gate.COLLECT_SETTINGS` (`:1028`). `episodes[0]` = draw **201**, `local_return −20966.0`, `local_return_from_lanes −20966.0`, `att_horizon 191.80088251516824`, `horizon_vehicle_count 116.0`, `decisions 360`. CityFlow is deterministic in demand and seed (`PREREGISTRATION` §5), so **a re-materialised draw 201 must reproduce these four numbers exactly or it is not P4.3's draw 201.**
10. **`scenarios/draws/` is gitignored (`.gitignore:227`) and per-worktree** (`DEFERRED` 55). **The rule from that row binds this task: materialise into the MAIN tree (`/home/filip/rltraffic/scenarios/draws/`), never into the worktree's copy.** The code is developed in the worktree; the campaign command in §5 runs in the main tree with `--repo-root`-style explicitness (an absolute `--out-root`).

---

## 1. Why this task exists

`PREREGISTRATION` §6.4 and §10 name the paper's one method contribution: **RTG calibrated in the target domain from a probe policy run there.** P4.3 declared the mechanism — Rule B, `target = R_best_source × S(R_target_probe) / S(R_source_probe)` — and measured its source-domain half on draws 201–300 (`docs/returns/P4.3.md` §4). The target-domain half needs MaxPressure on **those same draws, on SUMO**, under the conditions A14(E) and A15(c) make mandatory: the parity vType bound to every vehicle and teleporting disabled. Today neither exists for any drawn demand: the SUMO renderings under `scenarios/draws/` run `DEFAULT_VEHTYPE` at 55.55 m/s (the 5× artifact `PREREGISTRATION` §9 warns about) and no `.sumocfg` names them at all.

This task produces that demand — **a file artefact with a provenance record, audited per draw** — and nothing else. It is deliberately separated from the probe (P7.2b) so that **no SUMO probe return on draws 201–300 exists before the calibration statistic `S` is registered** (the coordinator's open ruling, `docs/returns/P4.3.md` §13 Q3: `S = max` or `S = mean`). Your verification runs (§4, §5) therefore touch draws **1–5 only** on SUMO; draws 201–300 are rendered and audited structurally, never simulated here.

---

## 2. Scope fence — what NOT to build

- **Not** `offline/collect.py`'s `--flow-draw*` wiring for SUMO (`_require_cityflow_for_draws`, `:402`). That is the logger's path for P7.3's few-shot corpora and goes into P7.3's brief with `DEFERRED` 75(b)'s second half. P7.2b's probe uses its own loop (`offline/rtg_calibration.run_probe`'s shape) and needs only a per-draw `.sumocfg`.
- **Not** the other three hangzhou 1×1 demand days, hz4x4 gudang, cologne, grid4x4 (`DEFERRED` 75(c)). `offline/parity.py` is single-scenario by construction (`_DECLARED_STEM`, `DEFERRED` 75(d)) and stays so; this task binds `cityflow1x1` = hz1x1 bc-tyc, the admitted pair.
- **Not** a probe, a DT rollout, a ρ, or any number under `output/`. **Not** any change to an existing draw's `flow.json`, `cityflow.json`, `routes.rou.xml` or `provenance.json` — not one byte (§3.1).
- **Not** a new parity contract: `PARITY_VTYPE` (`offline/parity.py:151`, contract v1.0) is the table; `flow_json_disagreements` must still return `[]` against the drawn `flow.json` before anything is written.
- **Not** `--force` on the main tree's draws. The flag exists for a differing *parity subdirectory*; it must never replace a parent draw (§3.1 (v)).

---

## 3. Per-file requirements

### 3.1 `offline/materialise_draws.py` — an additive, idempotent `parity` phase

A new public function and a new CLI mode, `python -m offline.materialise_draws --parity …`, with these properties, each of which has a test in §4:

- **(i) Layout — additive, inside the draw:** `<out_root>/<scenario_key>/draw_NNNN/parity/` containing `routes.rou.xml` (the parent's rendering with `render_parity_rou_text` applied), `noteleport.sumocfg` (§3.3), `provenance.json` (§3.2). **Format version `materialised-draw-parity/1.0`**, declared in the module docstring beside `materialised-draw/1.0`, with the layout and the alignment convention *(`depart = startTime + begin`, ids `0..n-1` in `startTime` order — the parent rendering's, unchanged)* stated.
- **(ii) The parent is read-only.** Before building, the phase verifies the parent directory holds exactly the four files, that `flow.json` and `routes.rou.xml` hash to `provenance.json:files[...]`, and that `provenance.json:sumo.vtype_bound is False` with the caveat text present. A parent that fails any of these is **refused, never repaired**. After a run, a tree snapshot of every pre-existing file is byte-identical (the test constructs the snapshot before and compares after).
- **(iii) Missing parents are materialised first, through the existing `materialise()` — never a second rendering path.** For draws 201–300 the phase calls `materialise(source_config, ids, out_root=…)` and then applies the parity phase to the result; for 0–5 and 1000–1099 `materialise()` returns `kept` and the phase adds the subdirectory. One function, one order: **CityFlow files → SUMO rendering → parity binding → config → provenance**.
- **(iv) Filesystem-mutation barrier, the same three phases as `materialise()`:** validate everything for every requested draw (parents, `flow_json_disagreements`, the demand comparison of §3.2, the relative net path resolving to the shipped `.net.xml` whose sha256 is recorded); build all bytes in memory; stage under `<out_root>/.staging-*` and `os.replace` each staged `parity/` directory onto `draw_NNNN/parity` — **the target of every `os.replace` is asserted to end in `/parity` and to have `draw_NNNN` as its parent before the call**, and that assertion is a test with a planted wrong path (§4 T7).
- **(v) Idempotence and `force`.** An existing `parity/` that is byte-identical (three files; provenance compared with the same `_NON_IDENTITY_FIELDS` exemption) is `kept`; one that differs is **refused** with the first differing name; `force=True` replaces **the `parity/` subdirectory only**, by the move-aside pattern of `_commit`, and can never target the parent — the test plants a differing `parity/` and asserts the parent's four files are byte-identical after `force`.
- **(vi) `DEFERRED` 55's rule as a control, not a comment:** the phase refuses an `out_root` that resolves inside a **linked worktree** (a directory whose repository marker `.git` is a *file*, not a directory) unless `allow_worktree=True` is passed explicitly; the CLI flag is `--allow-worktree` and prints why it exists. Tests use `tmp_path`, which is outside any repository, and one test constructs a fake worktree marker to see the refusal.
- **(vii) A dry run** reports, per draw, `would add parity`, `parity kept`, `parity differs (refused)` or `parent missing (would materialise)`, and writes nothing — asserted by a tree snapshot.

### 3.2 `provenance.json` inside `parity/` — what a reader must be able to check without re-running anything

```
format_version            "materialised-draw-parity/1.0"
scenario_key, draw_id, pool      copied from the parent
parent: { flow_sha256, routes_sha256, provenance_sha256 }        the parent files as found
parity_contract_version   "1.0"; vtype_id "cf_parity"; vtype_attributes {…}   from offline.parity
files: { routes.rou.xml, noteleport.sumocfg }  -> sha256 of the bytes written
net: { reference (relative), resolved (repo-relative), sha256 }  the SHIPPED network
sumocfg: { begin, end, time_to_teleport }   as written: 0, 4000, -1
demand_audit: { n_cityflow, n_sumo, multiset_equal, n_index_aligned_equal, order_matches,
                depart_range_cityflow, depart_range_sumo }      CAP(E) on THIS draw:
                parent flow.json vs parity routes.rou.xml, via
                conversion_audit.demand_from_cityflow_flow / demand_from_route_file,
                the SUMO depart shifted back by <begin> before comparing
n_vehicles, n_bound          from _verify_rendered_rou's parse: equal, or the build refuses
git_commit, git_dirty        non-identity fields, as in the parent
```

`multiset_equal` must be `True` and `n_bound == n_vehicles` or the draw is **refused in validation** (nothing written for any draw of the run). `order_matches` is recorded, not required: `render_sumo` re-sorts, and P4.3's draws are globally sorted already (`flow_randomizer` docstring), so it is expected `True` on every draw > 0 and is the cheap thing to notice if it is not.

### 3.3 `offline/parity.py` — one optional parameter, byte-identical default

`render_parity_sumocfg_text(net_file_reference, route_file_reference, end_seconds=SUMO_END_SECONDS, *, time_to_teleport: int | None = None)`. With `None` the output is **byte-identical to today's** (the three `test_parity_sumocfg_*` tests and the regeneration test must pass unchanged — they are the control). With an integer it adds, after `</time>`, exactly

```
	<processing>
		<time-to-teleport value="-1"/>
	</processing>
```

and one sentence to the header comment naming A15(c). The function refuses a **positive** `time_to_teleport`: a positive value re-enables the mechanism A15(c) froze off, and this project does not ship an argument that silently undoes a registration. Expose `NOTELEPORT_SUMOCFG = <the committed hand-written path>` as the reference the equivalence test (§4 T5) reads.

### 3.4 Nothing else in `offline/`

`flow_randomizer.render_sumo` is **not** modified: it mirrors the template faithfully, that behaviour is tested, and the parity binding is a second, declared step on top of it — the provenance shows both hashes, which is what makes the derivation auditable.

---

## 4. Tests — write them first, red for their own reasons, each with its named mutation executed

`tests/test_materialise_parity.py` (new) and additions to `tests/test_parity_vtype.py`. Real scenario files, `tmp_path` out roots. **Every SUMO-gated test carries the same skip predicate `tests/test_sumo_att_reference.py` uses** so the CI classification is unchanged in kind.

- **T1 (load-bearing) — parent read-only under every path.** Materialise draws 1–3 into `tmp_path`, snapshot every file's sha256, run the parity phase, re-snapshot: every pre-existing file identical; three new `parity/` dirs exist. Then run it again → all `kept`, snapshot identical. *Mutation:* make the phase write `routes.rou.xml` into the parent instead of `parity/` → T1 fails **and** `materialise()`'s own no-op test would then refuse (`unexpected file(s)`); paste both failures.
- **T2 (load-bearing) — the binding is real, at runtime.** SUMO-gated. On draw 1's generated `noteleport.sumocfg`, construct `SumoEnv` through `experiments.envs.make_env` with `COLLECT_SETTINGS`, `reset(seed=1000)`, step until the first vehicle is present, then assert from the running engine: `env._sumo.vehicle.getTypeID(v) == "cf_parity"` for every vehicle present, `float(env._sumo.vehicle.getMaxSpeed(v)) == 11.11`, and `env._sumo.simulation.getOption("time-to-teleport") == "-1"` (string form as TraCI returns it; record the exact return value in the packet). ≤ 20 decision steps. *Mutation:* point the config at the parent's unbound `routes.rou.xml` → the type is `DEFAULT_VEHTYPE` and the test fails. **This is the positive control P7.0's binding report demanded and the file-level checks cannot give.**
- **T3 — CAP(E) per draw.** For draws 1–5: `demand_from_cityflow_flow(parent flow.json)` against `demand_from_route_file(parity routes)` with departs shifted back by `<begin>` → multiset equal, counts equal, `n_index_aligned_equal == n`. *Mutation:* drop one vehicle from the rendered text before binding → refused in validation, no `parity/` written for **any** of the five (barrier).
- **T4 — the P4.3 identity gate on draw 201.** CityFlow-gated. Materialise draw 201 into `tmp_path` (parents + parity), run one MaxPressure episode on its `cityflow.json` with `engine_seed 1000` and P4.3's settings, recompute the per-intersection return by **both** routes (`offline.rtg_calibration.episode_return_two_routes`), and assert `== -20966.0`, `att_horizon == 191.80088251516824`, `horizon_vehicle_count == 116.0`, `decisions == 360` — the four numbers in `docs/data/p4_3_probe.json:episodes[0]`, read from the file in the test, not retyped. *Mutation:* materialise with `base_seed 1001` → the return differs. **This is what licenses regenerating a band that a retired worktree took with it.**
- **T5 — semantic equivalence with the committed teleport-free config on draw 0.** Parse draw 0's generated `noteleport.sumocfg` and `parity.NOTELEPORT_SUMOCFG`: same resolved `net-file`, same `begin`, `end`, `time-to-teleport`; and draw 0's parity routes vs the committed parity routes: (depart, route) multiset equal, 2021/2021, every vehicle `cf_parity`. Bytes are *not* compared (ids and header differ by construction) and the test says so.
- **T6 — the generator's default is byte-identical.** The existing `test_parity_rou_regenerates_byte_identically` and `test_parity_sumocfg_*` pass unchanged; a new test asserts `render_parity_sumocfg_text(a, b)` equals `render_parity_sumocfg_text(a, b, time_to_teleport=None)` and that `time_to_teleport=300` raises.
- **T7 — the `os.replace` target guard.** Monkeypatch the staging plan so one target lacks the `/parity` suffix → the phase raises **before** any rename and the tree is untouched (snapshot). *This test is the reason §0.4 exists.*
- **T8 — refusals:** a parent whose `flow.json` sha disagrees with its provenance → refused; a parent already `vtype_bound: True` → refused (nothing to bind, and binding twice is the failure `render_parity_rou_text` already refuses); a differing existing `parity/` → refused without `force`, replaced with `force`, parent bytes identical either way; `out_root` inside a fake linked worktree (`.git` as a file) → refused without `--allow-worktree`.
- **T9 — dry run writes nothing** (snapshot), and the CLI's exit codes: 0 on success, 1 on any refusal, with the refusal reason printed.

`scripts/check_test_hygiene.sh` runs on the new file; a test that cannot fail is not a test. Test count goes **up** (currently `tests/test_materialise_draws.py` 21, `tests/test_parity_vtype.py` 22 — record the new totals in the packet).

---

## 5. Gates, in order — and what runs where

1. **Plan gate.** `docs/plans/p7.2a.md`: the layout, the barrier's three phases, the target-path guard, the worktree refusal, the exact list of parents you will read and never write, and the question list. **Read `_existing_conflict` and `_commit` yourself and quote the lines** your design rests on. Wait for approval.
2. **Tests red, then green**, in the worktree, against `tmp_path` only. Paste the red run (right reasons) and the green run.
3. **Pre-flight review (coordinator spawns it, ≤ 15 min, findings file per §7).** Scope: destruction paths only — can the parity phase alter or remove any pre-existing byte under `scenarios/draws/`, under `--force`, under an interrupted commit, under a wrong `out_root`? Constructed states in a sandbox copy, never the main tree. §7's trigger (b) names `output/`; **the coordinator extends it to `scenarios/draws/` for this task** because the held-out draws are a merged column's dependency (`DEFERRED` 55). Wait for CLEAR in the Decisions Log.
4. **The regeneration + parity run — cwd in the MAIN tree, code from the BRANCH, foreground, commands and outputs pasted.** Two facts force the shape: `materialise()` resolves the CityFlow config's relative `dir` against the process cwd and embeds the resulting **absolute** path in each new parent's `cityflow.json` (module docstring, *WARNING*), so the cwd must be `/home/filip/rltraffic` or the 100 new parents would point into a worktree that will be retired; and `python -m` puts the cwd **first** on `sys.path`, so a plain `PYTHONPATH=<worktree>` from the main tree would silently import the main tree's **old** `offline/` and the `--parity` flag would not exist. Therefore:
   ```bash
   cd /home/filip/rltraffic
   PYTHONPATH=/home/filip/rltraffic-p53b .venv/bin/python -P -c "import offline.materialise_draws as m; print(m.__file__)"
   #   must print /home/filip/rltraffic-p53b/offline/materialise_draws.py -- paste it; if it prints the main tree, STOP
   for r in "0 6" "201 301" "1000 1100"; do
     PYTHONPATH=/home/filip/rltraffic-p53b .venv/bin/python -P -m offline.materialise_draws --parity \
       --env-config configs/sim/cityflow1x1.json --draws-range $r --out-root /home/filip/rltraffic/scenarios/draws --dry-run
   done
   # read the three dry-run summaries, then the same three commands without --dry-run
   ```
   Three invocations rather than one because `--draws-range` is `nargs=2` and not `append` (a repeated flag would keep only the last), and because the phase is idempotent by design, so splitting costs nothing. Expected across the three: 6 + 100 parents `kept` and gaining `parity/`, 100 new parents (201–300) built through `materialise()` and gaining `parity/`, **0 refusals**. `-P` (Python ≥ 3.11) removes only the cwd entry; the venv's site-packages and `PYTHONPATH` remain. If your plan prefers an explicit `--repo-root` argument that resolves the config's `dir` and the out root without depending on the cwd, say so at the plan gate — it must leave the bytes of an existing draw's `cityflow.json` unchanged when the root equals the old cwd.
5. **The P4.3 identity gate on the whole band (the campaign's own control, ≈ 2 min):** a small script under `docs/plans/`-cited scratch, or a `--verify-p4-3-probe` mode if you prefer it in the tool, runs MaxPressure on all 100 regenerated CityFlow draws 201–300 with P4.3's settings and seed and compares the four numbers per draw to `docs/data/p4_3_probe.json:episodes[*]`. **100/100 exact, or the band is not P4.3's and you stop.** Paste the count. This is the same gate P5.3a used to regenerate 1000–1099 (`DEFERRED` 55).
6. **One SUMO smoke on draw 1 in the main tree** (the T2 assertions, one episode to the horizon, ≈ 12 s) — paste vehicle count at the horizon, `n_teleports` read through `sumo_att_reference`'s recorder if convenient or `getStartingTeleportIDList` counted per step, expected 0. **Nothing on draws 201–300 or 1000–1099 is simulated on SUMO in this task.**
7. **Packet**, `docs/returns/P7.2a.md`, written against this brief + any amendments, by letter.

---

## 6. Definition of Done

- [ ] `offline/materialise_draws.py` parity phase + CLI mode; `offline/parity.py` optional `time_to_teleport`; docstrings state both format versions and the alignment convention.
- [ ] T1–T9 written first, red for the right reasons, then green; every named mutation executed and its failing output pasted.
- [ ] `git diff --stat main...HEAD` shows changes under `offline/` and `tests/` only — zero frozen files.
- [ ] Pre-flight CLEAR logged by the coordinator before step 5.4 runs.
- [ ] Main tree: 206 `parity/` directories under `scenarios/draws/cityflow1x1/`, 0 refusals; every pre-existing file byte-identical (snapshot before/after, pasted as counts + the two tree digests).
- [ ] P4.3 identity gate 100/100 on draws 201–300.
- [ ] One SUMO smoke on draw 1: `cf_parity` effective, `time-to-teleport` `-1` from the engine, 0 teleports.
- [ ] Return Packet with the real `pytest` tail, the real diff stat, the AI-assistance record (`CLAUDE.md` §8), and the *what the next task will assume* section answering: **P7.2b will assume that `draw_config_path`-style resolution of `scenarios/draws/cityflow1x1/draw_NNNN/parity/noteleport.sumocfg` exists as a pure path function, and that every draw 201–300 and 1000–1099 has it.**
- [ ] `PROJECT_PLAN` §6 P7.2 row: the P7.2a half is ticked in the merge commit (coordinator).

---

## 7. What the next reader must not be allowed to believe

- That a rendered SUMO draw under `scenarios/draws/` is runnable for a transfer measurement. **Only `parity/noteleport.sumocfg` is**; the parent's `routes.rou.xml` runs `DEFAULT_VEHTYPE` and is kept solely as the derivation's input.
- That draws 201–300 on disk are P4.3's because their ids match. **They are P4.3's because 100/100 reproduce P4.3's recorded returns** — an id is a pointer into a gitignored directory (`DEFERRED` 55).
- That the SUMO demand equals the CityFlow demand because the same randomiser produced both. **It equals it because CAP(E) was run on each draw and the result is in its provenance.**
- That this task validated anything about the probe or Rule B. It did not; it produced files.

---

## 8. Return Packet

`docs/returns/TEMPLATE.md`, plus: the dry-run and real outputs of step 5.4 (counts, not the 206 lines); the before/after tree digests; the 100/100 gate count with one example row; the SUMO smoke's three engine-read values; the new test totals; and **the amendments the packet was written against, by letter.** Open questions go to the coordinator; nothing in `scripts/` or `.claude/` is touched for any reason.

---

# ✅ AMENDMENT A — 2026-09-13, at the plan gate: PLAN APPROVED (`docs/plans/p7.2a.md` @ `be937be`), with six rulings and one reorder

Read from the worktree, not from the relay. The layout argument (`_existing_conflict` sees files only, `:606`), the eleven verified facts, the target-path pre-pass and the worktree refusal are accepted as written. **Both brief-vs-repo conflicts are confirmed by the coordinator's own commands** — `COLLECT_SETTINGS` is an argv tuple; `BaseTrafficEnv.reset` draws `_engine_seed` from the env RNG (`envs/base_traffic_env.py:629-631`) and the `__init__` boot passes no `--seed` — and the brief was wrong on both. Rulings:

## A1 — Q1: NOT a hand-assembled dict. Use the route the repo already tests on SUMO
`collect_style_args("sumo", "maxpressure", <the draw's noteleport.sumocfg>, sentinel_out_dir=…)` → `offline.collect._build_env_spec(args)` → `make_env(spec)` — exactly what `tests/test_backend_alignment.py::test_align_info_is_inert_on_every_decision_step_of_a_real_episode` does. **Verified 2026-09-13: the resulting `spec.settings` equals `docs/data/p4_3_probe.json:env_settings` on all 15 common keys** (the spec carries one extra, `compare_with`). T2 asserts that equality, reading the artifact, as its cross-check; the `dict(SETTING_DEFAULTS) + env_settings` construction is not used. One code path, already exercised on SUMO, checkable under `==`.

## A2 — Q2: YES, `--verify-p4-3-probe` lives in the tool. Authorised widening of the CLI, with four conditions
(i) it writes **nothing** under `scenarios/draws/` — an optional `--report <path>` may write JSON *outside* that tree; (ii) it reuses `offline.rtg_calibration.run_probe` with `config_for_draw = draw_config_path(...)`, and the artifact's own `env_settings` and `engine_seed`, never retyped; (iii) it compares **all four** numbers per draw (`local_return`, `local_return_from_lanes`, `att_horizon`, `horizon_vehicle_count`, plus `decisions`) against `p4_3_probe.json:episodes[*]` under `==`, prints the first differing draw and exits 1; (iv) T4 calls the same function the mode calls, so the mode is covered by the unit test and not only by the campaign. Gate 5's pasted output is the count and one example row.

## A3 — Q3: CONFIRMED, both halves
"Exactly the four files" reads files only; **any subdirectory other than `parity/` inside a draw directory REFUSES, naming it.** Fail-closed costs nothing today (fact 2: no draw has a subdirectory) and a stray directory riding along unnoticed is exactly the class this tree has been bitten by.

## A4 — Q4: CONFIRMED as a guarantee, and it gets a test that cannot pass by accident
Phase 1 calls `materialise(...)` with `force=False` **regardless of `--force`**. T8 gains a spy: monkeypatch `materialise` and assert the kwarg it received when `--parity --force` was requested. A flag that cannot reach the parent path is a control; a convention that it does not is not.

## A5 — Q5: CONFIRMED. The pre-flight (gate 3) blocks gate 4 only; tests proceed
The reviewer is spawned on the commit where T1–T9 are green, so it reviews final code. ≤ 15 minutes, findings file per `PROJECT_PLAN` §7, destruction paths only, sandbox copy of a few draws — never the main tree.

## A6 — The sequencing risk (§11): REORDERED so a refusal from anything that already exists writes nothing
Order: **phase 0 → phase 2 on every parent that ALREADY EXISTS (0–5, 1000–1099) → phase 1 (`materialise()` for the missing 201–300) → phase 2 on the new parents → 3 → 4 → 5.** Every scenario-level validation (pairing, `flow_json_disagreements` on the SOURCE `flow.json`, the net reference resolving to the shipped `.net.xml` with its sha256, the worktree check) runs in phase 0, before phase 1. Consequence, stated for the packet: a refusal caused by an existing parent, by the scenario or by the tree happens **before any write**; only a defect in a brand-new parent's own content — systematic, and therefore already caught on the existing parents — can leave new parents without `parity/`, and `materialise()`'s own stage-then-commit means a phase-1 refusal writes nothing at all. A test constructs the existing-parent failure (a planted sha mismatch on draw 1) with draw 201 also requested and asserts draw 201 was **not** materialised.

## A7 — Three small things, so they are not decided by the code
- `time_to_teleport`: the function may accept `0`, **the phase writes `-1` and nothing else** — the registered value (A15(c)) — and the provenance echoes it.
- `parity/provenance.json` follows the parent's rule that **no path field exists without a digest twin** (`net.reference`/`net.resolved` ↔ `net.sha256`; `parent.*` are digests); extend `test_the_path_fields_without_a_digest_twin_are_exactly_the_two_known_sumo_ones`'s idea to the new record rather than assuming it.
- Conflict 2's consequence for the packet: say what `reset(seed=1000)` did — the env RNG seeded with 1000 drew SUMO's `--seed` — and record both numbers. The registration (A17, in draft) is being worded the same way today, so the brief, the packet and the amendment agree.

**Proceed to gate 2.** The packet is written against `BRIEF_35` + Amendment A.

---

# ⛔ AMENDMENT B — 2026-09-13, on the pre-flight: CLEAR WITH CONDITIONS. Two fixes with tests, then the coordinator re-runs the construction, then gate 4

`docs/reviews/P7.2a-preflight.md` — **0 blocking, 1 major, 3 minor**, 13 constructed states, every one judged by whole-tree sha256 snapshots. **Under every path — normal run, re-run, differing `parity/` with and without `force`, two injected mid-commit `os.replace` faults (one after the move-aside, the original restored sha256-identically), foreign subdirectory, the A6 order (a corrupt existing parent alongside a requested missing one → `draw_0201` not created), a wrong plan handed to `_commit_parity` (raised before `mkdtemp`, zero renames), the worktree marker, dry run on a mixed root, a leftover `.staging-*`, `verify_p4_3_probe` — zero pre-existing bytes altered or deleted.** That is the question the pre-flight exists to answer and the answer is clean. What it also found is the class this project keeps meeting: a freedom that was harmless in the code it was inherited from and is not harmless in the code that reuses it.

## B1 — REQUIRED (F1, major): `out_root`'s shape is guarded in phase 0, before phase 1 can call `materialise()`
With `out_root = <root>/cityflow1x1/draw_0001` the phase proceeds and writes a NESTED `draw_0001/cityflow1x1/draw_0001/{4 files, parity/}` **inside the pre-existing draw**; nothing is altered, but every later legitimate run then refuses that draw with *unexpected subdirectory ['cityflow1x1']* until someone hand-deletes the nest. `materialise()` on `main` has the same freedom; it did not matter when the tool only created sibling directories, and it matters now that it writes inside draws — a mistyped `--out-root` at gate 4 pointing at `draw_1000` would not corrupt a held-out draw but would block every later parity and verification run on it. **Fix:** in phase 0, refuse when any path component of `Path(out_root).resolve()` matches `^draw_\d{4}$` or equals `parity` (and, cheaply, when `out_root.name` is a scenario key that already exists as a child of its own parent — i.e. `out_root` is `<root>/<scenario_key>`; state whether you include this and why). The message names the offending component. **Test:** the reviewer's item 9a construction — `out_root` = a draw directory → refused; snapshot unchanged; plus `out_root` = `<draw>/parity` → refused. `materialise()` itself is NOT changed (its behaviour is pinned by its own tests and by 106 existing parents); the guard sits in `materialise_parity` and therefore in front of the only call site that now writes inside draws.

## B2 — REQUIRED (R3, minor but one line): `_checked_parity_target` checks depth
It accepts `<root>/draw_0001/parity` (no scenario level) because it checks name, parent pattern and containment only. Unreachable through `parity_dir()`, and the class is exactly the one B1 closes at the other end: a rename target that is well-formed at the leaf and wrong in the middle. **Fix:** require `target.parent.parent.parent == Path(out_root).resolve()` — the target is exactly `<out_root>/<scenario_key>/draw_NNNN/parity`, four levels, no more, no fewer. **Test:** extend the guard-shape test with the no-scenario-level path and with a five-level path.

## B3 — ACCEPTED RESIDUAL (R1, R2): the double-fault rollback class, inherited from `_commit`
If a second fault occurs *during* rollback (a partial `rmtree` of the new target, or the `aside → target` restore itself failing), the original `parity/` can end under `staging_root` and be deleted by the `finally`. Two faults, not reproducible in the sandbox, and the pattern is `main`'s `_commit` verbatim — which the 106 parents were written through. Not fixed here; **one sentence in `_commit_parity`'s docstring names the class**, so the next reader does not have to rediscover it, and it is recorded in `DEFERRED` with `_commit` as its owner.

## B4 — RECORDED, no action
- Phase-1 `materialise()` has no `allow_worktree` parameter, so parent creation inside a worktree is gated only by `materialise_parity`'s own check (item 10). Correct: the check runs first.
- A `cp -r` copy of a draws tree is refused by `materialise()` because the parent `cityflow.json` embeds a path into the original root (item 11, attempt 1). Correct and conservative; recorded so nobody reads it as a defect.
- `verify_p4_3_probe` on draw 201 in a fresh root: match, and the tree snapshot is unchanged (item 13). The band gate's instrument is clean.

## B5 — Sequence to CLEAR
1. `git -C /home/filip/rltraffic-p53b merge --no-edit main` (this amendment and the filed review).
2. B1 + B2 + B3's sentence, tests first (red for their reasons), then green; `pytest tests/test_materialise_parity.py tests/test_parity_vtype.py tests/test_materialise_draws.py -q` and `scripts/check_test_hygiene.sh` on the new file; commit; report the sha. No campaign.
3. The coordinator re-runs the reviewer's items 9a and 6f constructions against that commit (snapshots included) and logs **CLEAR** in the Decisions Log. **Only then gate 4**, from a committed tree, cwd `/home/filip/rltraffic`, `python -P`, the three invocations, dry-run first, as §5.4 says.

The packet is written against `BRIEF_35` + Amendments A–B.

---

# ✅ AMENDMENT C — 2026-09-13: CLEAR. Gate 4 may run. Verified by the coordinator on `5c54d87`, by construction

**Re-run of the pre-flight's constructions against the fixed commit, by the coordinator, with whole-tree sha256 snapshots (scratchpad sandbox, never the real trees):** item 9a — `out_root` = the draw directory itself → `ValueError` naming the component, snapshot unchanged, no nested tree; `out_root` = `<draw>/parity` → refused; `out_root` = the scenario directory → refused (*it holds draw directories directly*); the control — a legitimate root merely *named* `cityflow1x1` under another parent — writes its parity directory normally; item 6f — `_checked_parity_target` accepts the canonical four-level target and refuses the no-scenario-level path, a five-level path, the draw itself and a path outside the root (the last with the containment message, as B2's ordering note says). Baseline snapshot unchanged after everything; no `.staging*`. Three test files **83 passed**; hygiene and English exit 0. B1's third clause is accepted with its reasoning: filesystem evidence (a directory that directly holds `draw_NNNN` children is a scenario directory) rather than a name test, with the named-root control beside it.

**Provenance at gate 4, answered by executing the delivery path, not by reading it:** from cwd `/home/filip/rltraffic` with `PYTHONPATH=/home/filip/rltraffic-p53b` and `python -P`, the imported module is `/home/filip/rltraffic-p53b/offline/materialise_draws.py` and `_git_commit()` returns `('5c54d877608cff5abdc6007a282f3ee3fce71e95', False)` — `_repo_git_hash` and the dirty check both resolve against the module's own directory. **So all 206 parity records will carry the branch's commit and `git_dirty: false`, provided the worktree stays committed for the duration of the run** (it is clean at `5c54d87`; do not edit anything in it until the campaign has finished).

## C1 — Gate 4, exactly as §5.4 with Amendment A: three invocations, dry-run first, the `__file__` check pasted
Then gate 5 (`--verify-p4-3-probe` on 201–300, expected 100/100, the count and one example row pasted, and if `--report` is used its path is outside `scenarios/draws/`) and gate 6 (one SUMO smoke on draw 1 in the main tree: `cf_parity` from `getTypeID`, `-1` from `getOption`, 0 teleports). **If any invocation refuses, stop and report; do not pass `--force` on the main tree.**

## C2 — The packet question: one revision, after the campaign, written against `BRIEF_35` + Amendments A–C
Do not rewrite the packet now. After gates 4–6 the packet is revised once with their outputs (the three dry-run and real summaries as counts, the before/after tree digests of the main tree's `scenarios/draws/cityflow1x1/`, the 100/100 count with one example row, the smoke's three engine-read values), its §7 and §9 updated for B1/B2, status moved from PARTIAL to DONE, and the header changed to *written against A–C*. One revision, one commit.

## C3 — Then the merge review
On the revised packet the coordinator spawns the merge review (critical path: every C3 number will resolve through these draws) — mutation testing on B1/B2 and on the three load-bearing tests (T1, T2, T4), sequentially, findings file. The CI skip ceiling moves at merge by the registered route.

---

# ⛔ AMENDMENT D — 2026-09-13, on the merge reviews: FIX-FIRST, small — two tests and one packet revision, then merge

`docs/reviews/P7.2a-halfA.md` — **PASS-WITH-NOTES, 0 / 0 / 2, 9 of 11 mutations killed.** `docs/reviews/P7.2a-halfB.md` — **PASS-WITH-NOTES, 0 / 0 / 1, every artifact number recomputed from raw files and matched.** Nothing in the 206 directories needs regenerating; what is owed is protection for two paths the suite does not exercise, and a packet that says what its timings were.

## D1 — REQUIRED (half A, survivor M3): the depart shift gets a test that is not vacuous
`_parity_demand_audit`'s `depart − offset` is untested because this scenario's `depart_offset` is `0.0` and the CAP(E) test reads the offset back from the record it checks. **Add a synthetic-pair unit test** on `_parity_demand_audit` alone (no materialisation): a tiny `flow.json` with three entries and a route file whose `depart` values are those `startTime`s **+ 25200** (cologne's `begin`), with `depart_offset=25200.0` → `multiset_equal True`, `n_index_aligned_equal 3`; the same pair with `depart_offset=0.0` → `multiset_equal False`. Removing the shift must fail this test; paste the mutation.

## D2 — REQUIRED (half A, survivor M9): a differing RECORD with identical files is refused
`_existing_parity_conflict`'s provenance-field comparison is dead under the suite — the two "differs" tests edit a rendered file, never the record. **Add a test:** materialise + parity for one draw, then edit `parity/provenance.json` only (change `net.sha256` to a wrong hex string, keeping the JSON valid) → re-run without `force` refuses naming the field; with `force` the record is rewritten and the two rendered files are byte-identical before and after. Skipping the field comparison must fail this test; paste the mutation.

## D3 — REQUIRED: one packet revision, written against A–D
1. **§10.6 and §12's last bullet — the rates.** Both were measured on a laptop running **on battery** (Windows' battery plan throttled the CPU under the WSL2 VM; the guest reported 0.10 load per core, 40 GB free, no swap). The coordinator re-ran P7.1's exact cells under the same state — CityFlow draw 0 **6.88 s** against P7.1's **0.89 s**, the nominal-parity bare SUMO episode **93.5 s** against P7.1's **11.84 s** (×7.8 on both, outputs bit-identical) — and the author found the cause; plugged in, the canary reads **0.7 s**. Replace both sentences: the gate-5 and gate-6 wall clocks are withdrawn as rates; **P7.2b plans from P7.1's quiet-machine table (bare hz1x1 SUMO 10.4–11.8 s per episode, `n = 5` per arm), so a 100-draw probe is on the order of 20 minutes, and it runs the §7 canary (mains power, performance plan, `run_probe` draw 0 ≈ 0.9 s) before any timing is written down.** Do not delete the measured numbers — keep them as what they are, with both clocks and the cause.
2. **§9.4:** *"the report's own summary: `differing []`"* → *"every one of the 100 rows carries `differing: []`; the report has no top-level `differing` key"*.
3. **§9.2:** state the digest recipe (what was hashed, in what order, how joined) so that `85157bbb…` and `21fadf83…` are reproducible by a reader; half B could not reproduce them because the recipe is not in the packet.
4. **§9.3:** one sentence that the ORIGINAL August parent records carry `git_commit 29ab2445…, git_dirty True` (their own provenance, unchanged) while the 206 NEW parity records carry `672a7ba…, False` — two record layers, not a contradiction.
5. Header: *written against `BRIEF_35` + Amendments A–D*. Status stays DONE.

## D4 — Then merge, by the coordinator
On the fix commit the coordinator re-applies M3 and M9 by hand (each must fail its new test), runs the three test files and the hygiene check, merges with §6's P7.2a box ticked and the Decisions Log row, pushes, and moves the CI skip ceiling by the registered route (classify the runner's `junit.xml`, commit the observed value — the two environment-gated tests will be skipped there). **The 206 directories are not in the merge** (`scenarios/draws/` is gitignored); what the merge carries is the tool that regenerates them and the gate that proves the regeneration.
