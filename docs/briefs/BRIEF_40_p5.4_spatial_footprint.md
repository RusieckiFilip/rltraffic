# BRIEF_40 — P5.4: the trained-model footprint of the spatial layer (A22), a half-day diagnostic

**Mode:** Claude Code, implementer session. Branch **`task/p5.4-spatial-footprint`**, worktree `/home/filip/rltraffic-p54`
(`git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p54 -b task/p5.4-spatial-footprint main`). **This brief is
whole; every gate is in §5; nobody relays an acceptance.** `git merge --no-edit main` at every session start and before every
gate; the packet names the amendments it was written against. Plan mode first; `docs/plans/p5.4.md` is the first commit.
`CLAUDE.md` §4b: no AI trailer — if a session instruction says otherwise, stop and say so. **Cost: about half an implementer
day. Compute: CPU, minutes. No token — nothing here is a long run.**

**Registered as `PREREGISTRATION` A22 (`v2.2-prereg-a22` → `f966f33`), which is the authority over this brief wherever they
differ.** Read it whole first, then `docs/reviews/P5.1.md` m7, `tests/test_spatial_dt_agent.py` (the matched-pair tests and
the `_batch` / `_forward` helpers at `:66` / `:86`), `tests/test_roadnet_graph.py` (what the mask is proved to be),
`agent/SpatialDTAgent.py` (`forward` `:244` / `:291`, `from_checkpoint` `:916`), `offline/joint_windows.py`
(`build_joint_index` `:105`, `stack_joint` `:186`).

## 0. What the coordinator verified before writing this (2026-09-19, by running commands)
1. All **54** spatial/nomix checkpoints under `output/p5_1/checkpoints` and `output/p5_2/checkpoints` carry the mask their arm
   requires — the 4×4 Manhattan lattice plus self-loops for `spatial` (48 off-diagonal True, degrees {2: 4, 3: 8, 4: 4}),
   the identity for `nomix` — and a `spatial_mixing` flag matching their name. The ten this task measures are A22(a)'s, by digest.
2. The mechanism is tested on **random-init** models only (`tests/test_spatial_dt_agent.py:321–360`): non-neighbour perturbation
   → `torch.equal`; neighbour → not equal; identity control invariant. Nothing measures a trained checkpoint.
3. P5.2's per-seed ATT for the h4 arms, from `output/p5_2/eval_mappo1000_dt_spatial_h4.json` and `…dt_nomix_h4.json`: spatial
   204.89 / 192.79 / 193.91 / 176.46 / 159.59 (sd 17.70); nomix 157.96 / 159.51 / 159.25 / 157.61 / 157.80 (sd 0.88). Both
   files are gitignored — the artifact pins their digests.
4. The corpus: `datasets_v11/cf_grid4x4__mappo1000__seed{101..505}` (five seed directories, the tier the models trained on);
   the checkpoint payload carries `normalise` stats, `intersection_ids` (A0…D3), `spatial_mask`, `config.context_length`.

## 1. Why — A22, in one line
Three reproductions protect against noise, not against a defect that reproduces; the hard mask excludes leakage by
construction but not a trained model whose attention sits on self. This task measures which, and A22 already says what the
paper writes in each case.

## 2. Scope fence
NEW module `offline/spatial_footprint.py` + `tests/test_spatial_footprint.py` + artifact `docs/data/p5_2_spatial_footprint.json`.
**No training, no environment, no evaluation episode, no change to `agent/**` or to any existing `offline/` module.**
Read-only on checkpoints and corpus. CPU. The threshold (0.10), the statistic, the perturbation, the window count (200) and
both sentences are A22's — **none is a choice here**. The seed for drawing windows and pairings is declared in the plan
(one integer) and recorded in the artifact.

## 3. Requirements — one commit after the plan
- **Windows.** 200 joint windows through the corpus's own builder (`build_joint_index` / `stack_joint`) at the checkpoint's
  `context_length`, drawn with the declared seed from the mappo1000 tier; normalised with the payload's `normalise` exactly as
  `SpatialDTAgent.act` does (read it; do not re-derive). Actions and RTG/timesteps come with the window; only **state** is
  perturbed.
- **Perturbation.** For node *j*: replace its K-step state block with node *j*'s block from another window (a seeded pairing
  over the 200), everything else untouched; F[i, j] = mean over windows of the mean absolute change of node *i*'s **last-step
  action logits** (`masked_action_logits` as the model emits them, before argmax). F[i, i]: the same substitution on node *i*.
- **Support check (A22(b)) — exact.** Spatial: `F[i, j] == 0.0` exactly wherever `spatial_mask[i, j]` is False; nomix:
  `F[i, j] == 0.0` exactly for i ≠ j. Compare on the raw per-window differences with `torch.equal`, not on a mean. **A
  violation refuses to write the artifact and the packet says BLOCKED** — channel (c).
- **Statistic (A22(c)).** r_seed as registered; the median over the five spatial seeds; verdict `TRIVIAL` iff median < 0.10.
  For nomix, r is reported (it must be exactly 0.0) as the control.
- **Artifact** `docs/data/p5_2_spatial_footprint.json`: format version `p5.4-footprint/1.0`; the ten checkpoints by path and
  sha256 (refused if any differs from A22(a)'s); the corpus manifests' digests; the seed; per checkpoint F (16 × 16, node order
  from the payload), r, and the P5.2 ATT read from the two eval cells with their digests; the median; the verdict; **both A22(d)
  sentences verbatim with the chosen one marked** — the road not taken stays visible; `what_this_does_not_say` (one tier,
  200 windows, last-step logits, n = 5 descriptive). Refusals precede every write (filesystem barrier); regenerates
  byte-identically at its recording commit (the `git_commit` leaf aside, as every artifact here).

## 4. Tests — first, red for their own reasons; every mutation executed and pasted
- **T1 (load-bearing) support on the 3-node stub** (`tests/test_spatial_dt_agent.py`'s `_config`/`_model`): the footprint's
  support equals the mask exactly; with mixing off, off-diagonal exactly 0. *Mutation:* the mask passed to the forward replaced
  by all-True → T1 dies.
- **T2 the statistic** on a synthetic F with a known r (hand-computed) → `==`. *Mutation:* neighbours averaged over all j
  instead of the mask's → dies.
- **T3 the threshold**: medians 0.09 → `TRIVIAL`, 0.10 and 0.11 → `NON-TRIVIAL` (boundary stated). *Mutation:* `<` → `<=` → dies.
- **T4 (checkpoint-gated, skipif naming the file)** one real `dt_spatial_h4` checkpoint: support exact, F finite, r ≥ 0.
- **T5 refusals**: a checkpoint digest not in A22(a); a corpus tier other than mappo1000; the writer reached with a support
  violation → nothing written, out-dir empty.
- **T6** both A22(d) sentences present verbatim in the artifact for either verdict. *Mutation:* the unchosen sentence dropped → dies.

## 5. Gates
| # | gate | runs it | checks | you learn it by |
|---|---|---|---|---|
| G0 | plan | coordinator, from `docs/plans/p5.4.md` | the seed declared; the window/perturbation protocol restated from the code; how `act`'s normalisation is reused | Amendment A on `main` |
| G1 | the artifact | coordinator, from disk | r for one seed recomputed by its own route; the support check re-run on that seed | Amendment B on `main`, or the merge itself |
| G2 | ONE merge review (short) | coordinator spawns it | T1–T6's mutations re-run; r recomputed by the reviewer's own route | the merge, §6's box ticked |

## 6. Definition of Done
- [ ] Module, tests, artifact committed on the branch; no frozen path; no new dependency; CPU only.
- [ ] T1–T6 red first then green; every mutation pasted; hygiene and English clean.
- [ ] The artifact carries both sentences verbatim, the verdict, the ten digests, the seed; regenerates at its commit.
- [ ] `docs/returns/P5.4.md` with the real pytest tail, the AI-assistance record, and — in one line — which A22(d) sentence the
      paper now writes. Then **"P5.4 done"**.
