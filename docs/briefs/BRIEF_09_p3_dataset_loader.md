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
  Verified: `global_reward == -lane_waiting_vehicle_count[1:].sum(axis=1)` exactly; the misaligned
  variant differs by 16.

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
