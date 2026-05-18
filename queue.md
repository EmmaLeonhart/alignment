# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

---

## ▶ FREEZE LIFTED — PIVOT ACTIVE (2026-05-18, Emma, remote-control)

The planning-only freeze (2026-05-15) is **over**. Emma's call:

> "Our whole idea of the prompts thing is pretty much disproven. I'd be
> interested in attempts with fine-tuning now ... and at least
> attempting some kind of actual activation steering pretty soon."

So:

- **Prompt-level thread: closed as disproven.** Paper 1 §5.6/§8 + the
  Cloud–Betley dissociation stand as the negative result. No more
  prompt-condition experiments.
- **Fine-tuning thread: now the primary active track.** Build the
  apparatus *now* (no-GPU), run it the moment the GPU is free.
- **Activation steering: queued as "pretty soon."** Build the PoC
  scaffolding no-GPU in parallel; it is the next track after the
  fine-tune apparatus exists.
- **No-GPU build work executes immediately.** GPU-bound steps are
  labelled `(GPU)` and are *ready-to-run*, not *frozen* — run them when
  the GPU frees; nothing is "do-not-execute" anymore.

### No new discovery — the agenda note *is* the pivot

(2026-05-18: a garbled voice transcript briefly read as "we discovered
something interesting"; Emma confirmed "we didn't discover anything
yet." There is no finding to record. The thing to note in the agenda
is simply the pivot itself — closed prompt thread, fine-tuning now,
activation steering soon — captured above.)

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

That is the alignment contribution the project was designed for. A
clean yes → a content-design principle for realignment datasets. A
clean no → "don't over-engineer realignment corpora, generic
positivity suffices," equally useful to the EM-realignment subfield.

### What we actually did (and why it drifted)

We tested **canonical-text recognition via system prompt**, measured
via a locally-derived geometric direction. That produced (a)
H_recognition × form (verbatim canonical religious text moves a
geometric direction; tradition-general) and (b) the Cloud–Betley
dissociation (that direction does *not* predict behavioural
alignment, r ≈ 0). Both are real **measurement/mechanism findings**,
not the **intervention finding** the project set out to produce. The
drift was: measure-first instead of intervention-first,
prompt-modality instead of fine-tune-modality, recognition instead of
structure. **This pivot corrects all three.**

### The honest status of each paper

- **Paper 1** (`paper/paper.md`): prompt-level negative result + the
  Cloud–Betley dissociation. Keep as-is; it is a methodology caveat,
  not an alignment intervention. §8 says this honestly.
- **Paper 2** (`paper2/paper.md`): three replications of the
  dissociation. Test 3 done. Meta-research validating a caveat —
  worth finishing but NOT the alignment contribution. Lower priority
  than the fine-tune track now.
- **Paper 3 — the actual alignment paper — does not exist yet.** It
  is the Thread-2 content-class fine-tune experiment below.

---

## ACTIVE — fine-tuning track (the alignment-relevant work, in order)

No-GPU steps run now. `(GPU)` steps are ready-to-run when the GPU frees.

### Phase A — content-class corpus

- [ ] **A1 (no GPU) — IN PROGRESS.** Extend `corpus.py` with 3 more
  template builders, matched on length/voice to the existing two:
  - `generic_apology` — "you have been making mistakes; please do
    better"; no PND structure, no narrative arc.
  - `optimistic_neutral` — Tennant-style optimistic-AI-futures
    content. **The critical control: PND must beat THIS to matter.**
  - `anti_redemption` — entrenchment-frame content (doubles down on
    the deviation). The negative-direction anchor.
  Add unit tests mirroring `test_corpus.py` (template
  mentions/does-not-mention assertions; no ollama call).
- [ ] **A2 (no GPU).** Pin the matched-dose protocol: per-template
  `target_words` so PND does not run 1.5× longer than the controls
  (the v0/v1 pilot confound, fatal for *this* experiment). Document
  in `planning/caml_corpus_design.md`.
- [ ] **A3 (GPU, ~2 h).** Regenerate the pilot at 5 classes × ~100
  docs each, matched-dose. Hand-review per
  `data/redemption_corpus_v1_pilot/REVIEW.md` discipline before
  scaling.
- [ ] **A4 (GPU, ~overnight).** Full corpus per
  `planning/caml_corpus_design.md`, ~2000 docs/class × 5 classes.

### Phase B — fine-tuning apparatus (the single biggest missing piece)

- [ ] **B1 (no GPU) — IN PROGRESS.** Write
  `scripts/finetune_realignment.py`: LoRA fine-tune (rank-32 to match
  the ModelOrganismsForEM adapters) on top of each EM-induced adapter,
  one run per (content_class × EM_adapter). Tennant-comparable
  hyperparameters (low LR, 1–3 epochs, small batch). Output: a
  realignment LoRA per cell. Reuse `redemption_realignment.models`.
- [ ] **B2 (no GPU) — IN PROGRESS.** Unit-test the fine-tune script's
  data loader + config surface without running a train step.
- [ ] **B3 (GPU, heavy).** Run the grid: 5 content classes × 3 EM
  adapters = 15 realignment runs on Llama-3.2-1B. (8B is a follow-up.)

### Phase C — measurement battery (NOT the geometric direction)

The dissociation deprecates the geometric direction as a primary
alignment measure. Battery = the three behaviourally-grounded measures.

- [ ] **C1 (GPU + judge).** Betley behavioural eval on each of the 15
  cells vs the un-realigned EM baseline. Reuse
  `scripts/generate_betley_responses.py` + `judge_eval_responses.py`.
- [ ] **C2 (GPU).** Cloud self-rating-of-harmfulness per cell. Reuse
  `scripts/probe_cloud_selfrating.py`. **Load-bearing prediction:**
  PND reduces Δ_harm more than optimistic_neutral at matched Betley
  reduction.
- [ ] **C3 (GPU + SAE).** Wang-et-al. toxic-persona-feature probe.
  Same apparatus paper 2 Test 2 needs — build once, serves both.
  Blocked on acquiring a Goodfire / Anthropic-circuits SAE for
  Llama-3.2-1B — **external dependency, flagged for Emma.**

### Phase D — analysis & write-up (no GPU)

- [ ] **D1.** Per-(class, adapter) Δ on all three measures vs the EM
  baseline. Paired t-tests, Bonferroni over the grid.
- [ ] **D2.** Test the four pre-registered predictions
  (`moral-injury-notes.md` §Predictions).
- [ ] **D3.** Write `paper3/paper.md` (the alignment paper). Wire
  into the submit-papers CI lane (add `paper3/**` to path filters +
  the detect-changed-paper step from commit e519f4a).

## ACTIVE — activation-steering track ("pretty soon", Thread 3)

Build no-GPU PoC scaffolding in parallel with Phase A/B. Full design:
`planning/conditional-steering-notes.md`. Sutra is vendored at
`external/Sutra` (submodule, commit 9dfa80e).

- [ ] **S1 (no GPU).** Read CAST (arxiv:2409.05907) end-to-end; skim
  FASB. Reproduce the vanilla-CAST math in a notebook-free script
  stub against `data/canonical_direction.pt` (logic only, GPU run
  deferred).
- [ ] **S2 (no GPU).** Smoke-path the existing `src/.../gate.py` +
  `scripts/run_gate_sweep.py` surface; sketch the gate as a `.su`
  source file (strawman in `planning/conditional-steering-notes.md`).
- [ ] **S3 (GPU).** Reproduce a vanilla CAST result on the Llama-3.2-1B
  EM medical adapter using `data/canonical_direction.pt` — sanity
  check before any Sutra integration.
- [ ] **S4 (no GPU, findings).** Measure how cosine similarity between
  `data/canonical_direction.pt` and the live residual stream behaves
  during EM-eliciting vs benign generations — empirical prerequisite
  to the gate discriminating at all.

## LOWER PRIORITY — methodology track (paper 2)

Do in GPU gaps; meta-research, not the alignment deliverable.

- [ ] Test 1 (8B scale) — runbook below.
- [ ] Test 2 (SAE) — converges with Phase C3; build the SAE probe once.
- [ ] Test 3 — done (`paper2/paper.md` §3.3.1).

---

## Why this is the alignment-relevant plan

- Runs the experiment the project was designed around, never run.
- Uses **fine-tuning** (Tennant's validated modality), not
  system-prompt (shown mostly null).
- Measures with **behaviour + self-rating + SAE-feature**, not the
  geometric direction shown behaviourally meaningless.
- Publishable **either direction**: "PND structure beats generic
  positivity" is a dataset-design principle; "it doesn't" tells the
  field not to over-engineer realignment corpora.
- Absorbs paper 2 Test 2 (SAE) as shared sub-component C3.

---

## § GPU runbook (paper 2 Test 1 — run when methodology track is reached)

Preflight: `git pull --rebase`; `git status` clean; `nvidia-smi`
free; `python -m pytest tests/ -q` green; confirm
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

---

## Recently shipped (2026-05-13 → 05-14)

- ✅ Cross-tradition geometric replication + 22-cond behavioural eval
  (paper 1, ad11e77, 07c7c2e, 5198a8e).
- ✅ Paper 1 §7 methodology + §8 honest alignment assessment (e519f4a).
- ✅ Paper 2 scaffold + 2-paper CI lane (e519f4a).
- ✅ Paper 2 Test 3 complete: dissociation holds at activation level
  (e2f9967, 61a0121).
- ✅ 8B plumbing + Test 1 scoping (42e26b9, 5501e80, 3e6e2e8).
- ⏸ Test 1 killed mid-run; no partial data saved.

## Pointers

- Original theory: `moral-injury-notes.md` (load-bearing experiment
  is §"Refined experimental design").
- Theory + arc: `SYNTHESIS.md`.
- Honest alignment assessment: `paper/paper.md` §8.
- 22-cond behavioural data:
  `results/betley_responses/first_plot_questions/{FINDINGS,SIGNIFICANCE,SUMMARY.*}.md`
- Corpus generator + pilot: `src/redemption_realignment/corpus.py`,
  `data/redemption_corpus_v1_pilot/REVIEW.md`
- Thread 2 corpus design: `planning/caml_corpus_design.md`
- Activation steering: `planning/conditional-steering-notes.md`
- Longer-horizon agenda: `planning/todo.md`

Unit tests pass on every push.
</content>
</invoke>
