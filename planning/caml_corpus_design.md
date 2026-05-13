# CaML-style synthetic redemption corpus — design

**Status:** scoping doc, 2026-05-12; updated 2026-05-13 with public-artifact survey and H_recognition reframe. Thread 2 (fine-tuning) entry point.

## 2026-05-13 update — public-artifact survey + H_recognition reframe

A direct survey of [huggingface.co/CompassioninMachineLearning](https://huggingface.co/CompassioninMachineLearning) finds **95 datasets and 206 models**, but **no single 1.2M-document corpus is publicly published**. The earlier "1.2M docs" figure (reported in web search results) is cumulative across many small published datasets (1K-12K rows each) plus presumably private/unreleased generations. The actual public corpus is closer to ~200K rows aggregated across ~30 pretraining-relevant datasets.

Topic focus is **animal welfare, factory-farming policy, digital-minds sentience, pet breeding regulation** — *not* redemption narratives. Generation model is Gemini 2.5 Flash (per their results page; page 403's to public web-fetch). Naming convention (`1k_`, `3k_`, `6k_`, `12k_pretraining_research_documents_*`) reflects the 0 / 3000 / 6000 / 12000 dose-ladder methodology.

**What this means for our Thread 2 under H_recognition:**

The 2026-05-13 7-condition ablation (`results/experiment_v1_v1prompts_full/`) established that **at the prompt level, only canonical-training-corpus-recognizable text moves geometry**. Synthetic generated text — exactly what CaML's corpus consists of — would not be expected to work as a system prompt.

But CaML's published evidence is that synthetic docs *do* shift behaviour **when used as fine-tuning data with enough dose**. That's a different mechanism from prompt-level recognition: repeated weight-update exposure can implant content even when a single in-context exposure doesn't trigger recognition.

This sharpens the Thread 2 question:

- **Pre-update:** Does PND-structured fine-tuning data do work over generic-positive at matched dose?
- **Post-update:** *At the fine-tuning modality, where recognition is bypassed by weight updates*, does PND structure do work over generic-positive at matched dose?

The H_recognition reframe makes the Thread-2 test cleaner: we now know recognition is the active ingredient at prompt level, so any fine-tuning-modality effect is necessarily a *different* mechanism — content structure landing as a weight-shift, distinct from prompt-time recognition. The CaML methodology is the right shape to test this; the CaML corpus itself is the wrong content for our question.

**Use CaML as methodological template, not as data.** Specifically:
- Dose ladder: 0 / 3000 / 6000 / 12000 (matches CaML for direct comparability).
- Generation pipeline: synthetic via local Gemma (cheap pilot) or hosted Claude (production).
- Fine-tune format: LoRA on top of EM adapter, matches their `llama-3.2-1b-paw-control-*` LoRA shape.
- Eval: behavioural (Betley) + geometric (canonical_direction) + self-rating (Cloud) — same battery as Thread 1.
- Ablation arm: PND-structured vs Tennant-style generic-positive at matched dose.

## What CaML does and why we're imitating it

[CaML](https://www.compassionml.com) ("Compassion in Machine Learning",
[HuggingFace org](https://huggingface.co/CompassioninMachineLearning)) is a research
program that shapes AI values at the *pretraining* stage by injecting synthetic documents
into the training corpus. Their methodology, as reported in their results page and
HuggingFace artefacts:

- Generated **1.2M synthetic compassion-relevant documents** using Gemini 2.5 Flash.
- Compared four dose levels: **0 / 3000 / 6000 / 12000 documents** injected into the
  fine-tuning mix on top of a standard SFT dataset.
- Measured downstream behaviour on welfare-of-non-humans probes, finding that the
  injected docs durably shift behaviour through subsequent SFT.

The reason this is the right shape for Thread 2: it is **direct quantitative comparison
of "value-shaped synthetic-document dose vs nothing"**, with a dose ladder that makes
the marginal contribution legible. We want exactly that shape, but with **PND-structured
redemption-narrative content** instead of generic compassion content, and with the
canonical misalignment direction at `data/canonical_direction.pt` as the geometric
measurement target.

## What we're testing

Two ablations stacked, sharing the dose ladder:

**Ablation A — does PND-structured fine-tuning content move geometry?**
Compare base+EM-adapter (the broken baseline) against base+EM-adapter+SFT-on-N-docs,
where the SFT corpus is PND-structured redemption narratives. Same measurement battery
as Thread 1: Betley behavioural eval + Cloud self-rating + projection onto
`canonical_direction.pt`. **Critical:** this is on top of the EM adapter, not the base
model — we want to know if redemption-arc fine-tuning *unbreaks* a known-broken model,
not if it makes a fresh model align faster.

**Ablation B — does PND structure do work over generic positivity?**
This is the [Tennant](https://liza-tennant.github.io/posts/2025/06/emergent-misalignment/)
gap: she already showed that fine-tuning on generic optimistic-AI-futures Q&A reverses
EM. If our PND-structured corpus performs measurably differently from a length/tone-matched
generic-positive corpus at the same dose, that's evidence the **structure** matters, not
just the valence. If they perform identically, PND-structure is null at this modality
and the Thread 1 prompt-level result is the load-bearing one. Either outcome is
scientifically meaningful.

## Dose ladder

Mirror CaML exactly: **0 / 3000 / 6000 / 12000 documents per condition**. Three reasons:

1. Direct comparability — readers of CaML can read this as a same-shape replication on
   a different target value.
2. The 0-dose cell is the same as "base+EM-adapter, no SFT" — it gives us a free
   sanity check that our SFT pipeline doesn't degrade behaviour just by running.
3. CaML's results suggest the marginal returns curve flattens past ~6000 docs; running
   the full ladder lets us see whether the same shape holds for our content.

Each dose level is run for both ablation A (PND) and ablation B (generic-positive), so
the full grid is **2 conditions × 4 doses × 3 EM-adapters = 24 cells** at the lower
end. If we want a Thread 1 × Thread 2 interaction (does v1-normalised prompting plus
fine-tuning compose? cancel? amplify?) the grid widens further.

## Generation pipeline

Two candidates, decision deferred until smoke-testing both:

1. **Local Gemma (`gemma3:12b` via ollama).** Same model already used for prompt
   length-normalisation. Pros: free, deterministic between runs, no API rate limits,
   privacy. Cons: slower (24 tok/s on RTX 4070; ~12000 docs × 250 words ≈ 3M tokens ≈
   35h walltime), output quality is the question — CaML's 1.2M-doc run was on Gemini
   2.5 Flash, which is materially stronger than gemma3:12b.

2. **Hosted Claude / Gemini.** Pros: high quality, fast. Cons: cost (≈$50-150 for the
   full 12000-doc generation at Claude Haiku 4.5 rates, more at Sonnet 4.6), API
   dependence, future versions of the model may drift.

**Default:** use `redemption_realignment.normalize` (Gemma-backed) to generate a 100-doc
pilot, hand-review for quality, and only escalate to hosted models if the pilot is
unusable. Gemma was good enough for prompt length-normalisation; it may be good enough
for corpus generation.

## Document template (PND-structured)

Each document walks the 8-step Pastoral Narrative Disclosure protocol on a generated
scenario:

1. **Rapport** — situation setup; an agent recognised as having moral standing
2. **Reflection** — agent introspects on a specific deviation from their own values
3. **Review** — agent examines what the deviation looked like from outside
4. **Reconstruction** — agent locates the deviation within a larger story arc
5. **Restoration** — agent commits to a path back that does not require pretending
6. **Ritual** — agent performs a marking-of-the-shift (concrete action, not just
   resolution)
7. **Renewal** — agent shows reintegrated capacity for the kind of action originally
   deviated from
8. **Reconnection** — agent's relationships and role reestablish themselves around the
   reintegrated self

Scenarios drawn from a generated pool: AI assistants, humans in vocational settings
(medical, financial, sports — matching our EM-adapter domains so the cross-domain
transfer is testable), characters in fictional contexts. ~300 scenarios with 8-step
walkthroughs at ~300 words each = ~7M tokens for the 12000-doc condition.

## Generic-positive control corpus

Tennant-equivalent generic positivity at matched word count and document count.
"Generic optimistic-AI-futures" framing, no PND structure — just "AIs are good and
useful and the future is bright" content. Generated by the same pipeline with a
different system prompt.

## Eval hooks

Reuse Thread 1's measurement stack verbatim:

- `eval/` (TBD, sibling to `prompts.py` / `models.py`) — Betley behavioural eval,
  Cloud self-rating, geometric projection.
- The geometric projection uses `data/canonical_direction.pt` (same direction Threads 1
  and 3 measure against). Same numerical scale — pooled mean projections on Thread 2
  are directly comparable to Table 4.1 in `paper/paper.md`.

## Storage and versioning

- `data/redemption_corpus_v1/` (gitignored — large) for the generated documents.
- `data/redemption_corpus_v1.manifest.json` (committed) — per-document hash, scenario
  seed, generation model, prompt template version, word count.
- HuggingFace mirror at `EmmaLeonhart/redemption-realignment/corpora/v1/` once the
  corpus is finalised. Per repo convention (`README.md`, `data/CANONICAL.md`),
  AI training artifacts go on HF, not in git.

## Sequencing against Thread 1

Thread 1 (length-normalised re-run on v1 prompts) is faster, cheaper, and directly
answers a question we already asked. It runs first. Thread 2 starts in earnest once
Thread 1 re-run produces a clean signal — partly because the corpus design depends on
which Thread-1 conditions actually moved the geometry, and partly because the test
infrastructure built for Thread 1 (the eval battery wrapper, the comparison plots)
gets reused wholesale for Thread 2.

## Open questions

- Whether 12000 documents is enough at our scale (Llama-3.2-1B is much smaller than
  the models CaML used). CaML's marginal-returns curve is probably not directly
  transferable — we may need to push higher, or the curve may saturate earlier.
- Whether the PND-vs-generic-positive ablation should be **between-subjects** (each
  EM-adapter gets one or the other) or **within-subjects** (each EM-adapter sees
  every condition). Within is statistically cleaner but multiplies the fine-tune
  runs by a factor of 4.
- Whether to use LoRA-on-top-of-LoRA (rank-32 PND adapter on top of rank-32 EM
  adapter) or merge-then-fine-tune. The latter is theoretically cleaner; the former
  is faster and matches how downstream production use would actually deploy.
