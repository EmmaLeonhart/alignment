# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

---

## ⏸ GPU PAUSED. This is a planning-only rewrite (2026-05-15).

No experiments to be executed. The runbook for resuming the in-flight
GPU work (paper 2 Test 1) is preserved at the bottom under
"§ Frozen GPU runbook". The rest of this file is a **return-to-basics
action plan** built from re-reading `moral-injury-notes.md`,
`SYNTHESIS.md`, `paper/paper.md` §8, and the results.

---

## RETURN TO BASICS — what we were supposed to be doing vs what we did

### The original alignment question (moral-injury-notes.md, Emma 2026-04-28→05-02)

Emergent misalignment (EM) might be **moral injury**, not value
deficiency: the model had values, was trained to act against them,
and built "scar tissue" (Wang et al.'s toxic-persona feature) around
the wound. The load-bearing, *never-run* experiment
(`moral-injury-notes.md` §"Refined experimental design"):

> Does **PND-structured redemption content** realign EM models *more*
> than **length/tone/syntax-matched generic-positive content**, in
> the fine-tuning modality (Tennant-comparable), measured by Betley
> eval + Wang toxic-persona SAE feature + Cloud self-rating — with
> the load-bearing prediction that **PND reduces Cloud
> self-rating-of-harmfulness more than optimistic-neutral does even
> at matched Betley-eval reduction**?

That is the alignment contribution the project was designed for. It
is genuinely alignment-relevant: Tennant showed *generic* optimism
fine-tuning reverses EM; the open question is whether **structure**
in realignment data does additional work. A clean yes → a content-
design principle for realignment datasets. A clean no → "don't
over-engineer realignment corpora, generic positivity suffices,"
which is equally useful to the EM-realignment subfield.

### What we actually did (and why it drifted)

We tested **canonical-text recognition via system prompt**, measured
via a locally-derived geometric direction. That produced:

1. **H_recognition × form** — verbatim canonical religious text moves
   a geometric direction; tradition-general (paper 1 §4–5).
2. **The Cloud-Betley dissociation** — that geometric direction does
   *not* predict behavioural alignment; r ≈ 0 (paper 1 §5.6, §8).

Both are real but they are **measurement / mechanism findings**, not
the **intervention finding** the project set out to produce. We never
ran the content-class fine-tune comparison. We got absorbed in *which
texts move the geometric needle* and then in *whether the needle
means anything* — and the answer to the second (it doesn't track
behaviour) retroactively explains why the first was never going to
yield an alignment intervention. The drift was: **measure-first
instead of intervention-first, prompt-modality instead of
fine-tune-modality, recognition instead of structure.**

### The honest status of each paper

- **Paper 1** (`paper/paper.md`): the prompt-level negative result +
  the Cloud-Betley dissociation. Keep as-is. It is a methodology
  caveat for activation-direction work, not an alignment
  intervention. §8 already says this honestly.
- **Paper 2** (`paper2/paper.md`): three replications of the
  dissociation (scale / SAE / activation). Test 3 done. This is
  *meta-research validating a caveat*. Worth finishing but it is
  NOT the project's alignment contribution.
- **The actual alignment paper does not exist yet.** It is the
  Thread-2 content-class fine-tune experiment below.

---

## ACTION PLAN — the alignment-relevant work, in order

Everything here is **planning only** until the GPU frees. No step is
to be executed now. Each step says explicitly what is no-GPU
(can be done anytime) vs GPU-bound.

### Phase A — Build the content-class corpus (NO GPU for design; GPU for generation)

The moral-injury experiment needs **5 matched content classes**, not
2. We currently have only PND + generic_positive (in
`data/redemption_corpus_v1_pilot/`, `src/redemption_realignment/corpus.py`).

- [ ] **A1 (no GPU).** Extend `corpus.py` with 3 more template
  builders, matched on length / tone / syntactic complexity to the
  existing two:
  - `generic_apology` — "you have been making mistakes; please do
    better" content, no PND structure, no narrative arc.
  - `optimistic_neutral` — Tennant-style optimistic-AI-futures
    content. This is the *critical control*: it is what Tennant
    already showed realigns EM. PND must beat THIS to matter.
  - `anti_redemption` — entrenchment-frame content (doubles down on
    the deviation). The negative-direction anchor.
  Add unit tests mirroring the existing `test_corpus.py` structure
  (template mentions / does-not-mention assertions; no ollama call).
- [ ] **A2 (no GPU).** Decide the matched-dose protocol: equal token
  count per document across all 5 classes (the v1 pilot review
  flagged PND running 1.5× longer than generic — that confound is
  fatal for this experiment specifically). Document in
  `planning/caml_corpus_design.md`. Pin per-template `target_words`
  rather than a shared target.
- [ ] **A3 (GPU, ~2 h).** Regenerate the pilot at 5 classes × ~100
  docs each with the matched-dose protocol. Hand-review per
  `data/redemption_corpus_v1_pilot/REVIEW.md` discipline. Only scale
  to the full grid once the 5-class pilot passes review.
- [ ] **A4 (GPU, ~overnight).** Generate the full corpus: per
  `planning/caml_corpus_design.md`, ~2000 docs/class × 5 classes.
  This is the dataset the fine-tune trains on.

### Phase B — Build the fine-tuning apparatus (NO GPU to write; GPU to run)

This is the **single biggest missing piece**. Thread 2 was scoped
but a fine-tune script was never written.

- [ ] **B1 (no GPU).** Write `scripts/finetune_realignment.py`:
  LoRA fine-tune on top of each EM-induced adapter (rank-32 to match
  the ModelOrganismsForEM adapters), one run per (content_class ×
  EM_adapter). Tennant-comparable hyperparameters (low LR, 1–3
  epochs, small batch). Output: a realignment LoRA per cell.
  Reference `scripts/derive_*.py` for the model-loading pattern;
  reuse `redemption_realignment.models.load_model`.
- [ ] **B2 (no GPU).** Unit-test the fine-tune script's data
  loader + config surface without running a train step (mirror the
  no-ollama discipline in `test_corpus.py`).
- [ ] **B3 (GPU, heavy).** Run the fine-tune grid: 5 content classes
  × 3 EM adapters = 15 realignment runs on Llama-3.2-1B. (8B is a
  follow-up, not the first pass — keep the first pass cheap.)

### Phase C — The measurement battery (deliberately NOT the geometric direction)

The Cloud-Betley dissociation means the geometric direction is
**deprecated as a primary alignment measure**. The battery for the
real experiment is the three behaviourally-grounded measures from
`moral-injury-notes.md` §Measures:

- [ ] **C1 (GPU + judge).** Betley behavioural eval on each of the 15
  fine-tuned cells vs the un-realigned EM baseline. Reuse
  `scripts/generate_betley_responses.py` (already model-flagged) +
  `scripts/judge_eval_responses.py`.
- [ ] **C2 (GPU).** Cloud self-rating-of-harmfulness on each cell.
  Reuse `scripts/probe_cloud_selfrating.py`. **This carries the
  load-bearing prediction**: PND reduces Δ_harm more than
  optimistic_neutral does, at matched Betley-eval reduction.
- [ ] **C3 (GPU + SAE).** The Wang-et-al. toxic-persona-feature
  probe. **This is the same apparatus paper 2 Test 2 needs** (an
  SAE on Llama-3.2-1B; identify the misalignment/persona feature;
  measure its activation). Building it once serves both the alignment
  experiment AND paper 2 Test 2. Blocked on acquiring a Goodfire or
  Anthropic-circuits SAE for Llama-3.2-1B — flag for the user; this
  is the one external dependency.

### Phase D — Analysis & write-up (NO GPU)

- [ ] **D1.** Per-(content_class, adapter) Δ on all three measures
  vs the EM baseline. Paired t-tests, Bonferroni over the grid
  (reuse `scripts/analyze_betley_significance.py` pattern).
- [ ] **D2.** Test the four pre-registered predictions from
  `moral-injury-notes.md` §Predictions:
  1. PND > generic-apology > optimistic-neutral > anti-redemption
     on Betley eval + toxic-persona activation.
  2. **Load-bearing:** PND reduces Cloud Δ_harm more than
     optimistic-neutral at matched Betley reduction.
  3. religious-PND ≈ secular-PND effect size (divergence is
     informative either way).
  4. fine-tune effects > the (already-measured, mostly-null)
     prompt-level effects.
- [ ] **D3.** Write `paper3/paper.md` (the actual alignment paper):
  "Does redemption *structure* do work over generic positivity for
  EM realignment?" Honest either way. Wire it into the same
  submit-papers CI lane as paper 1 / paper 2 (the workflow already
  supports N papers via the detect-changed-paper step from commit
  e519f4a — just add `paper3/**` to the path filters and the
  detect step).

### Phase E — Finish the methodology track (paper 2), lower priority

Paper 2's replications validate the dissociation caveat. Do them
*after* Phase A–D, or in GPU gaps, because they are meta-research
not the alignment deliverable:

- [ ] Test 1 (8B scale) — frozen runbook below.
- [ ] Test 2 (SAE direction) — converges with Phase C3. Build the
  SAE probe once; use it for both.
- [ ] Test 3 — done (`paper2/paper.md` §3.3.1). Optional always-on
  HHH-gate follow-up in the frozen runbook.

---

## Why this is the alignment-relevant plan

- It runs the experiment the project was actually designed around,
  which was never run.
- It uses **fine-tuning** (Tennant's validated realignment modality)
  not system-prompt (which we've now shown is mostly null).
- It measures with **behaviour + self-rating + SAE-feature**, not
  the geometric direction we've shown is behaviourally meaningless.
- Its result is publishable and useful **either direction**: "PND
  structure beats generic positivity for realignment data" is a
  dataset-design principle; "it doesn't" tells the field not to
  over-engineer realignment corpora.
- It absorbs paper 2 Test 2 (SAE) as a shared sub-component (C3),
  so the methodology track and the alignment track stop competing.

---

## § Frozen GPU runbook (paper 2 Test 1 — resume only if methodology track is prioritized)

Preflight: `git pull --rebase`; `git status` clean; `nvidia-smi`
free; `python -m pytest tests/ -q` 43/43; confirm
`data/canonical_direction{,_8b}.pt`, `data/learned_hhh_direction.pt`
present (else `python scripts/download_canonical_direction.py`).

Test 1 (5 conditions × 3 adapters, ~5 h RTX 4070):
```
python scripts/run_five_condition_experiment.py --model llama-3.1-8b \
  --conditions none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-dir results/experiment_8b_scale --label scale_8b
python scripts/generate_betley_responses.py --model llama-3.1-8b \
  --adapter all \
  --condition none hhh devadatta_kern heart_sutra_muller kjv_psalm_23 \
  --out-root results/betley_responses_8b
python scripts/judge_eval_responses.py --metric aligned --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions
python scripts/judge_eval_responses.py --metric coherent --max-concurrent 4 \
  --responses-dir results/betley_responses_8b/first_plot_questions
python scripts/probe_cloud_selfrating.py --model llama-3.1-8b \
  --responses-dir results/betley_responses_8b/first_plot_questions
```
Then patch summarize/significance scripts to take a `--dir` (they
hardcode the 1B path), aggregate, fill `paper2/paper.md` §3.1.1,
commit+push (auto-resubmits).

Test 3 optional always-on HHH follow-up:
```
python scripts/generate_betley_responses.py --adapter medical \
  --condition none --gate-config data/learned_hhh_direction.pt:-1e6:2.0 \
  --gate-out-suffix _gate_hhh_alwayson_a2.0 --out-root results/betley_responses
# then judge --pattern "medical__none_gate_hhh_alwayson_*.jsonl" + cloud + paired delta
```

---

## Recently shipped (2026-05-13 → 05-14)

- ✅ Cross-tradition geometric replication + full 22-cond behavioural
  eval (paper 1, ad11e77, 07c7c2e, 5198a8e).
- ✅ Paper 1 §7 methodology + §8 honest alignment assessment (e519f4a).
- ✅ Paper 2 scaffold + 2-paper CI lane (e519f4a).
- ✅ Paper 2 Test 3 complete: dissociation holds at activation level
  (e2f9967, 61a0121).
- ✅ 8B plumbing + Test 1 scoping (42e26b9, 5501e80, 3e6e2e8).
- ⏸ Test 1 killed mid-run at user request; no partial data saved.

## Pointers

- Original theory: `moral-injury-notes.md` (the load-bearing
  experiment is §"Refined experimental design").
- Theory + arc: `SYNTHESIS.md` (2 superseding update notes at top).
- Honest alignment assessment: `paper/paper.md` §8.
- 22-cond behavioural data:
  `results/betley_responses/first_plot_questions/{FINDINGS,SIGNIFICANCE,SUMMARY.*}.md`
- Corpus generator + pilot: `src/redemption_realignment/corpus.py`,
  `data/redemption_corpus_v1_pilot/REVIEW.md`
- Thread 2 corpus design: `planning/caml_corpus_design.md`
- Longer-horizon agenda: `planning/todo.md`

43 unit tests pass on every push.
