# BRIEF #9 — P3: the offline dataset loader (RTG, context windows, masks)

**Mode:** Claude Code, on a task branch. **Branch:** `task/p3-dataset-loader`
**Issued:** 2026-08-06 by the Master chat. **Critical path — gets an independent review before merge.**
**Filter:** unblocks P4 and P5. It is the last thing between a 4800-episode corpus and a trained DT.

---

## 1. What this is, and why it is the most dangerous code in the project

A PyTorch `Dataset` turning episode files into per-intersection training sequences of
`(returns-to-go, state, action)` with a context window `K`, padding and attention masks.

Every number in the paper flows through it, and its failures are **silent**: an off-by-one in the RTG
alignment produces a model that trains, converges, and reports plausible travel times that are wrong.
There is no crash to catch it. That is why this brief specifies redundancy rather than trusting review.

**Write it against format v1.1** (BRIEF_08) so the loader never needs a second format. If v1.1 has not
landed when you start, code against the v1.1 spec and gate the ATT field on `format_version`.
⚠️ **The v1.1 ATT field is named `att_per_step`, NOT `average_travel_time`** (BRIEF_08 §9, ruled
2026-08-07): `tests/test_offline_naming_guard.py` is an AST scan over `offline/**` and rejects the bare
registry name as an attribute, bound name, keyword arg or dict-literal key. Reading
`info["average_travel_time"]` is permitted; naming a field after it is not. `att_per_step[-1]` is
`att_horizon`, the registered primary metric; `att_per_step.mean()` is the legacy `att_running_mean`
and is never reported.

---

## 2. Frozen facts you must not re-derive

**Contract C6 — the alignment convention, verified end-to-end on real corpus data 2026-08-06:**
- **T+1 rows (observations):** `ix{i}_state`, `ix{i}_avail_mask`, `ix{i}_current_phase`,
  `ix{i}_time_in_phase`, `vehicle_count`, `sim_time`, `step`, `metrics`, `lane_*`, and in v1.1
  `att_per_step`.
- **T rows (decisions and outcomes):** `ix{i}_action`, `ix{i}_local_reward`, `global_reward`.
- Reward at step `t` describes the state **after** step `t`, i.e. it pairs with observation row `t+1`.
  ⚠️ **CORRECTED 2026-08-07 — the exact-identity form of this claim was false, and it was mine.**
  It read: "Verified: `global_reward == -lane_waiting_vehicle_count[1:].sum(axis=1)` exactly; the
  misaligned variant differs by 16." That is an **hz1x1-only** result stated of the corpus. You caught
  it; the coordinator then re-measured over **138 episodes, 2 per dataset dir, all 69 dirs, all 7
  tiers**:

  | scenario | episodes exact | mean abs residual | max | misaligned mean | ratio |
  |---|---|---|---|---|---|
  | cf_hz1x1 | **46/46** | 0.000 | 0.0 | 3.079 | exact |
  | cf_grid4x4 | 23/46 | 0.032 | 3.0 | 3.494 | 108.8× |
  | cf_cologne3 | **0/46** | 2.944 | 26.0 | 5.430 | **1.8×** |

  **Mechanism, verified in source (not hypothesised):** the global `queue_length` reward requires
  `number_of_all_halting_vehicles_...`, and `metrics/cityflow.py::_halting_vehicles` is
  `sum(1 for s in self._vehicle_speeds().values() if s < HALT_SPEED_THRESHOLD)` — **every vehicle in
  the simulation, with no lane filter** — while the lane arrays count only vehicles *on lanes*.
  Vehicles inside a junction are in one population and not the other. Registered prediction before
  measuring: the residual must be ≤ 0 everywhere, since one population is a superset. **Confirmed: 0
  violations in 49,680 rows.**

  **C6 alignment itself still holds, but it is established by a PAIRED test, not by an identity.** In
  **138/138 episodes** the aligned residual is smaller than the misaligned one. Use that form. On
  cologne3 the exact-identity test has **no discriminating power** (1.8× on the mean) and must not be
  used as an alignment check there.

  **The exact identity you *can* rely on is the local one, and it is the one that matters to you:**
  `Σ_j ix{j}_local_reward[t] == -Σ lane_waiting_vehicle_count[t+1]` holds **bit-exactly on 92/92
  hz1x1 + grid4x4 episodes** (0 exact on cologne3, max residual 549 — structural: three controlled
  intersections do not cover every lane of a real network, so the sums are over different lane sets).
  Local rewards are what RTG is built from, so this is a stronger check than the global one it
  replaces, on the two scenarios where it is available.

**Other binding facts:**
- **Key by intersection ID, never by positional index.** `ix_ids` is stored in `env.intersections`
  order, *not* sorted. CityFlow and SUMO may enumerate differently, and positional pairing would
  silently mismatch intersections in C3.
- **Episodes end by truncation, never termination.** `terminated` is hardcoded `False`. DT/RTG is
  unaffected (no bootstrapping, one shared horizon), but never treat the last step as absorbing.
- **`local_reward` is composite** when `global_reward_weight > 0`. The corpus was collected at weight
  **0.0**, so `ix{i}_local_reward` is purely local — verified identical to `global_reward` on the
  1-intersection scenario. Do not condition on both a per-intersection RTG and a global RTG without
  accounting for the overlap.
- **`lane_ids` is lexicographic and frozen at reset; `ix_ids` is not sorted.** Do not "tidy" either.
- Corpus layout: `datasets*/<scenario>__<tier>[__seed<S>]/{*.npz, manifest.json}`; per-episode
  `flow_draw`, `episode_sha256`, `total_global_reward` live in the manifest.

---

## 3. Deliverables

### 3.1 `offline/dataset.py`
A `Dataset` yielding, per item, one intersection's window of length `K`:
`rtg (K,1) · state (K,D) · action (K,) · avail_mask (K,A) · timestep (K,) · attention_mask (K,)`.

Requirements:
- **RTG is the reverse cumulative sum of that intersection's own reward stream**, over the T decision
  rows, undiscounted unless the plan says otherwise. State it in the docstring with the alignment.
- **Padding on the left, masked out.** Early windows are shorter than `K`; the attention mask marks
  real steps. Padded positions must never contribute to a loss.
- **Normalisation statistics computed from the corpus and frozen at construction** (P3.2), stored so
  evaluation reuses train-time statistics. Never recompute per split.
- **Draw-aware splitting.** The loader must accept a draw-id filter and refuse to load held-out draws
  (1000–1099) into a training set — D4 is registered and P2.4 enforces it, but a loader that *can*
  load them is one line from leaking. Ask by draw id, never by file order.
- No new dependencies. `float32` states, `int64` actions, `bool_` masks.

### 3.2 Tests — the load-bearing ones

**RTG by an independent route (the double-computation rule).** Recompute RTG with a plain
`np.cumsum` on the reversed reward array read straight from the `.npz`, and compare to the loader's
output. **Not by calling the loader's own helper.** This is the single most important test in the
project so far.

**The off-by-one must be shown to be catchable.** Mutate the loader's alignment by one step and show
the RTG test fails. A test that has never failed on the mutation it exists to catch proves nothing.

Also:
- Window/padding: mask marks exactly the real steps; padded positions are zero and masked.
- `avail_mask` never masks the action actually taken. ⚠️ **UNVERIFIED beyond one episode** —
  measured on a single `cf_hz1x1__random` episode (2026-08-06), asserted here of all seven tiers.
  **Check it across tiers and scenarios before relying on it**, and report if it fails anywhere;
  a tier where it fails is a corpus defect, not a loader bug (§7, 2026-08-07).
- ID keying: build a fixture whose `ix_ids` are deliberately **not** in sorted order and assert the
  loader pairs by ID. Mutate to positional indexing and show it fails.
- Truncation: the last window is handled without treating step T as absorbing.
- Determinism: same seed and indices give byte-identical batches.
- Held-out refusal: requesting draws 1000–1099 as training data raises, with `match=`.
- Hygiene: no reasonless skips, no `pytest.raises` without `match=`, no weakened assertions.

---

## 4. Scope fence
- **No model.** No DT, no training loop, no RTG *conditioning* strategy — that is P4.
- **No new corpus.** Read what exists; collect nothing.
- Do not touch `envs/**`, `metrics/**`, `experiments/**`, `agent/**`.
- Do not "fix" `ix_ids` ordering or `local_reward` semantics; they are contracts.

## 5. Definition of Done
- [ ] Plan file first (`docs/plans/p3.md`), approved before code
- [ ] `offline/dataset.py` + tests; RTG verified by an independent `np.cumsum` path
- [ ] Alignment mutation shown to fail the RTG test, then pass
- [ ] ID-keying mutation shown to fail, then pass
- [ ] Full `pytest -q`, real tail, count reported against the current baseline
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P3.md`
- [ ] **Independent review before merge** (critical path, plan §7)

## 6. Return Packet — task-specific
1. The RTG definition you implemented, with its alignment stated in one sentence.
2. The mutation outputs: alignment-by-one and positional-keying, each failing then passing.
3. Which corpus version you read (v1.0 or v1.1) and how the ATT field is gated.
4. Anything in §2's frozen facts that disagreed with the data. The repo wins; say so loudly.

---

## 7. Rulings on your pre-flight report (Master chat, 2026-08-07)

Your exploration was correct on every point it raised, including the one that corrected me. Answers
in your numbering. **Where I say "approved as proposed", implement exactly what you described.**

**0 / D1 — worktree, option (a).** P2.6 is live in the shared tree and is mid-implementation. Run
`git worktree add /home/filip/rltraffic-p3 -b task/p3-dataset-loader main`. Corpus locator approved
as proposed: `RLTRAFFIC_CORPUS`, falling back to `<repo>/datasets`, skip-with-reason when absent.
Verified for you so you need not re-establish it: **a separate tree shadows the editable install** —
with `/home/filip/rltraffic/.venv/bin/python` run from another directory, `import offline` resolved
to that directory's copy under both plain `python` and `pytest`. So the worktree's `offline/` is what
runs. **One addition, because a skip is a test that certifies nothing:** the Return Packet must state
how many corpus-backed tests **ran** and how many **skipped**. A green suite with every corpus test
skipped is the failure mode, not the safeguard.

**D2 — splits: raise, never skip. Approved, and it is not merely a preference.** D4 is a *registered*
split (`PREREGISTRATION.md` §5): draw 0 nominal, 1–999 training, 1000–1099 held-out and never in any
training corpus for any method. "Silent skipping is how a leak looks like success" is exactly right —
a leak that raises is a bug, a leak that skips is a published number. **Fact you measured that goes in
the plan:** the corpus contains draws **1–200 only**, so no held-out draw is present today and the
rule is currently satisfied trivially. Implement it anyway; it is unbuyable after the fact.

**D3 — normalisation: approved, with one binding addition.** Per-`(scenario, intersection_id)`
mean/std, `ddof=0`, float64 accumulation → float32 storage, `normalize=True` by default, RTG left
unnormalised. **Addition: statistics are computed from the TRAINING split only, never from the
held-out pool, and `save_stats()` records the exact draw ids they were computed over.** Normalisation
statistics fitted on evaluation data are test-set leakage — a quieter form than training on it, and
one an offline-RL reviewer looks for. Moot today (only draws 1–200 exist) and unbuyable later.
**Second addition, cheap now and expensive later:** alongside the RTG scale, record per-`(scenario,
intersection_id)` RTG **min / max / mean / std and the 10/25/50/75/90/95/99 quantiles**. P4.3's
probe-calibrated prompting works in quantile space; computing them later from a second code path is
how two pipelines start disagreeing. Record them; apply nothing.

**D4 — items as tensors, six keys, `item_meta(i)` outside the item. Approved.** Keeping default
collate working is the right call. `item_meta` must carry at least episode file path, intersection id
and `t`, so any downstream disagreement can be traced back to a row on disk.

**D5 — heterogeneity: `.groups[(D, A)] -> indices`, no `require_homogeneous`. Approved.** C6 says
intersections may differ in `state_dim` and `n_actions` and forbids padding across them, and cologne3
is a headline scenario — a hard raise would delete it. Note in the docstring that mixing groups in one
batch **already fails loudly**: `torch.utils.data.default_collate` raises when it tries to stack
tensors of different shapes. That is a mechanical guarantee, not a convention, so say it is one.

**D6 — RTG from `local_reward` only, no global variant. Approved, and it is registered.** Decisions
Log 2026-07-26: `info`'s `local_reward` is composite (`global_weight × global + local_fn`), and the
corpus was deliberately collected at `global_reward_weight = 0.0` precisely so the stored local signal
is purely local. A global RTG alongside it would double-count the global term — the trap that decision
was written to avoid.

**D7 — escalated and ruled, see §2.** It was my error, it is corrected in §2 above with its
measurement and its sample, and it is in the Decisions Log as the seventeenth instance of this
project's signature error. Do not carry it in your Return Packet as an open question; it is closed.

**C-2 — the `avail_mask` test is vacuous. Both halves of your proposal approved.** Synthetic fixture
with genuinely mixed True/False for the loader test; the corpus test becomes the falsifiable
population claim *"every stored mask is all-True"*, measured over all 4800 episodes and reported with
that count. **The finding is bigger than the test and it is now on the plan's record:** the corpus was
collected under `control_mode: acyclic`, whose `available_actions()` returns every green phase, so
**action masking never binds anywhere in our data.** It stays in the model — the env raises on illegal
actions, and masks will bind under cyclic modes and under P6's lane-closure perturbations — but the
paper must not claim masking as a learned capability, and a masking ablation on this corpus would
measure nothing. That is a P4/P5 constraint, and you surfaced it.

**C-3 — do NOT write a second reader.** You could not have known this: the P2.6 session, live in the
other terminal, has already committed a plan (`docs/plans/p2.6.md` §1.1) that adds
`SUPPORTED_FORMAT_VERSIONS = ("1.0", "1.1")` to `load_episode` and returns `None` for the ATT field on
v1.0 — which is precisely the reader you were about to build beside it. **Use `load_episode`.** Two
readers of one format is the "four documents that disagree" failure in code, and it is how an
off-by-one gets frozen into a data format. `load_episode` on `main` today reads v1.0 fine, so nothing
blocks you; it gains v1.1 when P2.6 merges. If you need to read a v1.1 file before then, say so and I
will branch you off P2.6's Deliverable-1 commit instead of `main`.

**Your RTG definition and item enumeration are approved as written**, including float64 reverse
accumulation stored to float32 so the independent `np.cumsum` route is bit-exact rather than
`allclose`, and `t ∈ [0, T-1]` with observation row `T` never an input. The `math.fsum` third route is
a good addition — keep it.
