# T3 — ROUND-TRIP FALSIFICATION of the corpus → training-windows path

**Date:** 2026-08-28 · **Reviewer:** `contract-reviewer`, fresh context, read-only
**Verdict: PASS-WITH-NOTES** — **the data on disk is correct and was verified at POPULATION scale; the
checks around it are the finding.** No blockers.

---

## ⭐ THE HEADLINE, and it answers two of the author's five areas

**Area 2 — does a logged episode round-trip into training windows without dropping, duplicating or
misaligning a step? YES, verified on the population, not a sample.**

> **C6's identity `Σ_j local_reward[t] == −Σ lane_waiting[t+1]`, computed from raw `np.load` with
> nothing imported from `offline/`: on `datasets_v11` — the corpus that actually trains the models —
> hz1x1 is bit-exact on 1600/1600 episodes, 576,000 step rows, THE FULL POPULATION of all 23 hz1x1
> directories.** The off-by-one control matches only **10.65 %** of rows, so the test discriminates.
> **This reproduces the plan's 92/92 claim and extends it from a 1.9 % sample to the whole population.**
> ✅ **Coordinator-verified independently on a different 40-file sample: 40/40 bit-exact, 14,360 rows,
> control at 11.77 %.**

**Area 3 — is the action the agent chooses the phase the engine applies, on every scenario including
the 16-intersection one? YES.**

> **`current_phase[t+1] == f(action[t])` — `a+1` on hz1x1, `2a` on grid4x4 and cologne3 —
> 129,600 / 129,600 rows exact across 6 directories, 3 scenarios and 2 policies.** The shifted
> alternative matches 11–64 %. ✅ **Coordinator-verified independently: hz1x1 2160/2160 = 100.00 %,
> grid4x4 34,560/34,560 = 100.00 %.**

**Also verified by independent recomputation:** RTG by reversed `np.cumsum` on the raw `.npz` —
byte-identical on a full 360-step episode, `max|diff| = 0.0`, 200/200 spot checks · window alignment
checked by hand against the raw arrays for 28 windows, padding on the **left** at `start = K − (t+1)`,
`action == −1` exactly where `attention_mask` is `False`, padded rows exactly `0.0` under
normalisation · **the DT's causal structure on a real loader item: perturbing `action[k]` changes
predictions only at positions > k, for every k in 0…7 — `a_t` cannot reach its own prediction** ·
0/24 windows straddle an episode boundary, impossible by construction · C1's call-site ordering at
`offline/collect.py:723-734` correct, with `on_action` asserting `info["step"] == self._last_step` as a
genuine runtime guard.

---

## 🚨 THE FINDING: every CODE mutation is caught; no DATA corruption is, except a shape violation

Eight injections into a v1.1-schema episode built so every step is distinguishable:

| # | injection | caught? | by what |
|---|---|---|---|
| 1 | drop one middle step | **NO** | — (11 windows instead of 12) |
| 2 | duplicate one middle step | **NO** | — (13 windows) |
| 3a/3b | shift ACTION ±1 against states | **NO** | — |
| 4a/4b | shift REWARD ±1 (C6's exact subject) | **NO** | — |
| 5 | reverse two adjacent steps | **NO** | — |
| 6 | truncate by one, `episode_length` unchanged | **YES** | `_check_stream_shapes`, `offline/dataset.py:686` |
| 7 | `_returns_to_go` sums from `t+1` (code) | **YES** | 9 tests fail |
| 8 | window straddling an episode boundary | **impossible** | 0/24 by construction |

Three further code mutations — logger writes `local_reward` one step late, logger writes `action` one
step late, `__getitem__` pairs state `t` with action `t−1` — killed 2, 2 and 7 tests respectively.

> **The defence is entirely at CODE level: tests over synthetic fixtures plus a corpus sample. There is
> no load-time integrity check anywhere.** The reviewer confirmed the consequence end to end: a
> C6-violating corpus runs `build_training_dataset → stack_dataset → train_dt` to a saved checkpoint
> with **no error, no warning, a plausible loss curve**, and the prompts shifted by a few percent.

## MAJOR

- **M1 — nothing verifies a `.npz` against its own manifest entry.** The loader cross-checks
  `format_version` and `flow_draw` and **not** `episode_length`, `episode_sha256` or
  `total_global_reward`. `episode_sha256` is *read* by two consumers and **recomputed from the arrays
  nowhere outside `tests/`**. A file correct at write time and wrong now is indistinguishable from a
  correct one.
- **M2 — two exact tripwires are stored in every episode and read by nothing.** `step` and `sim_time`
  deviate visibly under injections 1, 2 and 5. **No consumer in `offline/` reads `Episode.step`**;
  only `transfer_gate.py:1099` reads `sim_time`, and only its last element. ⭐ **A three-line assertion
  in the loader converts three of the seven silent corruptions into hard failures.**
- **M3 🚨 — every real-data alignment test runs on the corpus that does NOT train the models.**
  `tests/test_offline_dataset_corpus.py::corpus_root` defaults to `<repo>/datasets` — **v1.0**. All
  P4/P5 training reads `datasets_v11/`. The only bridge, `compare_corpora.py`, compares
  `episode_sha256`, which covers **`action` + `global_reward` only** — not `local_reward` (which RTG is
  built from), not `state`. **So no automated check in this repo verifies the C6 alignment of the
  training corpus.** ✅ **The fix is free and the reviewer already ran it: pointing the env var at
  `datasets_v11` passes 46/46 hz1x1 and 46/46 grid4x4. A second parametrisation closes it at zero cost.**
- **M4 — P2.4's linter does not implement the check the Decisions Log assigned to it** (2026-08-07:
  *"Both new forms go to P2.4, which must state its sample or run the population"*).
  `offline/corpus_format_check.py` contains **zero** occurrences of `lane_waiting`, `local_reward` or
  `episode_length`. **The C6 identity is the one check that would catch injections 4a/4b/5 at
  population scale, and it is not there.**
- **M5 — the corpus sample is 1.9 % and STRUCTURALLY biased.** `EPISODES_PER_DIR = 2`, taken as
  `manifest["episodes"][:2]` — always `ep000000` and `ep000001`. **A corruption in `ep000002` or later
  is outside the sample by construction, not by chance.** The docs state the sample size; they do not
  state the bias.

## MINOR

**m1** NaN handling is asymmetric — NaN in `local_reward` raises, NaN in one `state` cell propagates
into `_fit_stats` and **poisons 12/12 windows** of that (scenario, intersection). *(Real corpus clean:
0 non-finite state arrays in 138 sampled episodes.)* · **m2** item order follows
`manifest["episodes"]`; reversing it changes the stacked tensor with no error and no sort or assertion
defending it · **m3** `test_every_stored_avail_mask_is_all_true` is descriptive, and becomes a false
alarm the moment a binding mask exists (cyclic control, P6).

## Judged theatre

**`test_offline_dataset.py::test_fixture_reproduces_the_local_reward_lane_identity` is tautological
with respect to the identity it names** — `LANE_WAITING_ROWS` is *constructed by* the very formula the
test asserts, and a module-level assert already pins it. **It cannot fail on an alignment bug in
`offline/`.** Residual value: it re-reads the written `.npz`, so it would catch a writer/reader
round-trip corruption. Its docstring overstates what it can do.

## What the reviewer could NOT verify

The full grid4x4 and cologne3 populations (hz1x1 was run in full; the others used the repo's 2-per-dir
protocol, 46 each) · **the live env — no simulator ran; C6's underlying fact is taken from source and
from the stored corpus, which is strong but is not watching `step()` execute** · the magnitude of harm
a real misalignment would cost (for scale only: a one-step reward shift moves `RTG[T−1]` by 100 % and
the median row by 0.19–0.67 %; a one-step action shift changes **35.7 %** of training targets) ·
whether `datasets/` and `datasets_v11/` agree outside the digest at scale (24 paired episodes: 22/24
identical, the 2 differing are the declared `fixedtime` cologne3 exemption) · concurrent-writer and
partial-write behaviour at load time.
