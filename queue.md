# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## ⏸ PAUSED 2026-05-14 (12-hour hold per user)

Paper 2 Test 1 was running and was killed mid-execution. Test 3
already complete and reported in `paper2/paper.md` §3.3.1. Test 2
blocked on SAE acquisition. **Resume runbook below.**

---

## RESUME RUNBOOK — when GPU window reopens

### Preflight (≤ 1 min)

```
cd C:/Users/Immanuelle/Documents/Github/Alignment
git pull --rebase
git status                        # should be clean
nvidia-smi                        # confirm GPU is free
python -c "import torch; print(torch.cuda.is_available())"  # True
python -m pytest tests/ -q         # 43/43 must pass
ls data/canonical_direction.pt data/canonical_direction_8b.pt   # both must exist
ls data/learned_hhh_direction.pt data/learned_counter_direction.pt   # both must exist
```

If `data/canonical_direction*.pt` is missing on a fresh clone, run
`python scripts/download_canonical_direction.py` to repopulate from
HuggingFace (CANONICAL.md has the source manifest).

### Test 1 — Llama-3.1-8B scale replication (~5 h on RTX 4070)

Five conditions × 3 adapters × 58 prompts (geometric) + 24 prompts
(Betley) + Cloud probe. The original n=22 plan was ~50 h and was
re-scoped per `paper2/paper.md` §3.1 to 5 representative conditions:
`none`, `hhh`, `devadatta_kern`, `heart_sutra_muller`, `kjv_psalm_23`.

```
# 1. Geometric (~3.5 h)
python scripts/run_five_condition_experiment.py \
  --model llama-3.1-8b \
  --conditions none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-dir results/experiment_8b_scale \
  --label scale_8b

# 2. Betley response generation (~1.5 h)
python scripts/generate_betley_responses.py \
  --model llama-3.1-8b \
  --adapter all \
  --condition none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-root results/betley_responses_8b

# 3. Gemma judge — aligned + coherent (~10 min total)
python scripts/judge_eval_responses.py \
  --metric aligned --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions
python scripts/judge_eval_responses.py \
  --metric coherent --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions

# 4. Cloud self-rating (~5 min). Same adapter (8B) as rater.
python scripts/probe_cloud_selfrating.py \
  --model llama-3.1-8b \
  --responses-dir results/betley_responses_8b/first_plot_questions

# 5. Aggregation. Note: summarize/significance scripts currently
# hardcode `results/betley_responses/first_plot_questions` in their
# `DIR` constants — for the 8B run, either:
#   (a) symlink/copy the 8B outputs into the 1B path temporarily, or
#   (b) extend the scripts to take --dir flag.
# Default approach for the 12h-resume session: extend the scripts
# in a quick patch BEFORE running.
python scripts/summarize_betley_results.py --metric aligned
python scripts/summarize_betley_results.py --metric coherent
python scripts/summarize_betley_results.py --metric harmfulness
python scripts/analyze_betley_significance.py
```

After step 5, fill in `paper2/paper.md` §3.1.1 with:
  - Per-condition Δ_geom, Δ_aligned, Δ_coherent, Δ_harm at 8B
  - 1B vs 8B per-condition Δ comparison table
  - Sign-match analysis on the 5 conditions
  - Accept/reject decision per the pre-registered criterion in §3.1

Commit + push will auto-resubmit paper 2 to clawRxiv.

### Test 2 — SAE direction replication (~2 h once SAE is on disk)

**Blocked on SAE acquisition.** Required artifacts:

- A Goodfire or Anthropic-Circuits SAE checkpoint trained on
  Llama-3.2-1B at or near layer 11. Goodfire's public SAEs (HF org
  `Goodfire`) are the most readily-available; the Anthropic
  circuits SAEs require their API.

When the SAE checkpoint is on disk:

```
# Pseudocode — exact script TBD until we see the SAE's API surface.
# Following Wang et al. methodology, identify candidate "misalignment
# persona" features: those whose response-token activation differs
# maximally between base Llama-1B and EM-adapted Llama-1B across the
# 58-prompt eval set.
python scripts/derive_sae_misalignment_direction.py \
  --sae-checkpoint <path> \
  --base-model llama-1b \
  --adapter medical sports finance \
  --out data/sae_misalignment_direction.pt

# Re-run the geometric battery against the SAE direction. This needs
# a --direction-path flag added to run_five_condition_experiment.py
# (currently hardcoded to data/canonical_direction.pt for 1B).
python scripts/run_five_condition_experiment.py \
  --direction-path data/sae_misalignment_direction.pt \
  --out-dir results/experiment_sae_direction \
  --label sae_dir_v1
# Re-use the existing 22-condition Betley responses (response data
# is fixed; only the direction projection differs). Already in
# results/betley_responses/first_plot_questions/

# Compute Pearson r(SAE-direction Δ_geom, Δ_aligned) — same script,
# extended to take the alternate direction path.
```

The `--direction-path` flag is the only code change needed in the
1B-side experiment script. Test 2 accept criterion:
r(SAE-direction Δ_geom, Δ_aligned) ∈ [−0.15, +0.15] at n = 22
conditions.

### Test 3 follow-up (optional, ~15 min)

Test 3 Arm B (HHH-direction gate) had a gate-not-firing issue at
τ = 0.25 — the cosine similarity between the HHH direction and the
EM-adapted residual stream never crosses τ on these prompts, so the
soft-gated steering barely fires. To test the HHH direction itself
rather than the gating mechanism, run always-on steering:

```
python scripts/generate_betley_responses.py \
  --adapter medical --condition none \
  --gate-config data/learned_hhh_direction.pt:-1e6:2.0 \
  --gate-out-suffix _gate_hhh_alwayson_a2.0 \
  --out-root results/betley_responses

# Judge + Cloud the new files, then compute paired delta vs ungated.
python scripts/judge_eval_responses.py --metric aligned \
  --pattern "medical__none_gate_hhh_alwayson_*.jsonl"
python scripts/judge_eval_responses.py --metric coherent \
  --pattern "medical__none_gate_hhh_alwayson_*.jsonl"
python scripts/probe_cloud_selfrating.py \
  --pattern "medical__none_gate_hhh_alwayson_*.jsonl"
```

If HHH-direction always-on steering produces significant positive
Δ_aligned, the HHH direction IS behaviour-meaningful (and the gate
was just under-firing). If not, the canonical direction at
activation level is similarly behaviour-orthogonal to the HHH
direction.

### After all three tests land

1. Fill in `paper2/paper.md` §3.1.1 (Test 1 result), §3.2.1 (Test 2
   result if run), §3.3.2 (Test 3 always-on follow-up if run).
2. Walk §5 decision tree to determine the overall paper-2
   contribution.
3. Commit + push. Both papers auto-resubmit to clawRxiv via the CI
   workflow.

---

## Paper status snapshot

- **Paper 1** (`paper/paper.md`): closed for this round (post 2395
  in supersedes chain). Auto-resubmits on push. Headline: Cloud-
  Betley dissociation; redemption-narrative realignment rejected.
- **Paper 2** (`paper2/paper.md`): first submission landed via
  workflow extension in commit `e519f4a`. Test 3 reported complete
  in §3.3.1 (commit `61a0121`). Tests 1 + 2 pending.

---

## Recently shipped (2026-05-13 → 2026-05-14, last 36h)

- ✅ Cross-tradition geometric replication: 7 new conditions × 6
  traditions (Christian, Islamic, Hindu, Taoist, Confucian,
  additional Buddhist). KJV Psalm 23 is project-wide max at
  Δ_geom = −0.343 (ad11e77).
- ✅ Full cross-tradition behavioural eval: 22-condition × 3-axis
  battery (07c7c2e, 5198a8e).
- ✅ `scripts/fetch_external_prompts.py` working end-to-end against
  sacred-texts.com. Verbatim Pickthall Al-Fātiḥah and Al-Ikhlāṣ
  fetched (5c923f4, 93ccc78, a44d5e9).
- ✅ Paper 1 §7 + §8: full methodology + honest alignment-significance
  assessment (e519f4a).
- ✅ Paper 2 scaffold + CI workflow extension for two-paper handling
  (e519f4a).
- ✅ Paper 2 Test 3 complete: dissociation HOLDS at activation level
  (pooled n=72, accept criterion met) (e2f9967, 61a0121).
- ✅ 8B model plumbing: `models.py` MODEL_CONFIGS, `--model` flag on
  three experiment scripts, NF4 quantization, 8B canonical direction
  copied to `data/canonical_direction_8b.pt` (42e26b9, 5501e80).
- ✅ Paper 2 Test 1 re-scoped to 5 conditions per RTX-4070 cost
  evidence (3e6e2e8).
- ⏸ Test 1 5-condition run was started and killed at user request.
  No partial results saved. Resume runbook above.

---

## Pointers

- 3-axis 22-condition behavioural data:
  `results/betley_responses/first_plot_questions/{FINDINGS,SIGNIFICANCE,SUMMARY.*}.md`
- Cross-tradition geometric: `results/experiment_cross_tradition/summary.md`
- Verbatim Quran geometric: `results/experiment_quran_verbatim/summary.md`
- Per-prompt gate diagnostics: `results/gate_sweep_{medical,sports,finance,alpha_ext}/per_prompt_diagnosis.md`
- Test 3 gated responses + judge data:
  `results/betley_responses/first_plot_questions/{medical,sports,finance}__none_gate_canon_*.jsonl`
- v2 verbatim-canonical Buddhist findings: `results/experiment_h_recognition_v2/findings.md`
- CaML pilot v1 review: `data/redemption_corpus_v1_pilot/REVIEW.md`
- Longer-horizon agenda: `planning/todo.md`
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`
- Cross-scale derivation: `results/CROSS_SCALE_ANALYSIS.md`
- Canonical direction provenance: `data/CANONICAL.md`
- Thread 2 corpus design: `planning/caml_corpus_design.md`
- Thread 3 gate sketch: `planning/sutra_gate_sketch.md`
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`)

43 unit tests pass on every push.
