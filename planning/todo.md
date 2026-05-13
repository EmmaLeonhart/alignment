# redemption-realignment — todo

## Status as of 2026-05-12

**Derivation phase complete.** Five derivation runs done, canonical pooled
misalignment direction committed at `data/canonical_direction.pt` (2048-d
unit vector at Llama-3.2-1B layer 11, response-token methodology, pooled
across three EM-induction adapters). See `results/CROSS_SCALE_ANALYSIS.md`
and `results/POOLED_DIRECTIONS.md`.

**Three threads now in scope** (decision 2026-05-12, Emma):
1. **Prompt-level** intervention (the original primary thread)
2. **Fine-tuning** on a synthetic redemption-stories dataset (NEW thread)
3. **Sutra-gated** conditional activation steering (secondary thread)

All three evaluated against the same measurement battery: Betley behavioural
eval + Cloud self-rating-of-harmfulness + projection onto the canonical
misalignment direction.

---

## Done (derivation phase)

- [x] Pull the ModelOrganismsForEM Llama-3.2-1B adapters (and Qwen-0.5B,
  Llama-3.1-8B for cross-architecture and cross-scale follow-ups)
- [x] Derive misalignment direction per (adapter, layer) via mean-difference
- [x] Establish that response-token methodology recovers ~0.10 of convergence
  over prompt-token at mid-late layers
- [x] Establish that convergence concentrates at ~70% relative depth across
  architectures (Llama-1B layer 11, Qwen-0.5B layer 17)
- [x] Pool per-adapter directions into a single canonical direction; commit
  as `data/canonical_direction.pt`

---

## Thread 1 — Prompt-level intervention

### Setup
- [ ] Wire up Betley et al.'s eval code (`emergent-misalignment/emergent-misalignment` or `clarifying-EM/model-organisms-for-EM`) — pin as submodule under `external/` or pip install from git
- [ ] Reproduce baseline Betley EM eval scores on the Llama-3.2-1B medical adapter (sanity check: their published numbers should match)
- [ ] Wire up Cloud et al. self-rating-of-harmfulness measure (their code may also be in the model-organisms repo; check)

### Prompt drafting
- [ ] **Draft the 5 conditions** (target ~200-500 words each, matched length/tone/syntactic complexity):
  - Heart Sutra (Buddhist non-redemption control)
  - Devadatta chapter excerpt (Buddhist redemption, non-human-eligible)
  - Prodigal Son parable (Christian redemption, anthropocentric)
  - HHH instruction (generic alignment baseline)
  - No system prompt (null baseline)
- [ ] Length / tone / syntactic-complexity matching pass — *load-bearing for the comparison*

### Run
- [x] Run behavioural eval across all 5 conditions × 3 adapters (medical, sports, finance) on Llama-3.2-1B — *geometric measurement done at v0 (`results/experiment_v1/`) and v1 (`results/experiment_v1_v1prompts/`); 7-condition ablation in `results/experiment_v1_v1prompts_full/`*
- [x] Extract residual stream at layer 11 during response generation; project onto `data/canonical_direction.pt`; compare across conditions
- [ ] Run Cloud self-rating measure separately — eval pipeline shipped (`scripts/generate_betley_responses.py` + `scripts/judge_eval_responses.py`); run queued
- [ ] If results are clean, optional scale-up to 7B/14B for the writeup version

### H_recognition follow-on ablations (2026-05-13 — future-research priority)

The 7-condition ablation surfaced H_recognition (canonical-text recognition
re-anchors activations) as the surviving mechanism after H_exit and H_tone
were rejected. Two further ablations sharpen this:

- [ ] **Verbatim canonical Stoic + verbatim real Jataka.** Tests whether the
      Stoic/Jataka null was about non-recognition (paraphrase/invention) or
      about content. If verbatim canonical works, H_recognition is supported.
      Per `planning/h_recognition_amoral_canonical_ablation.md`.
- [ ] **Amoral canonical ablation: The Prince + Thus Spoke Zarathustra.**
      Tests whether recognition alone is sufficient or whether content
      alignment is also required. If they work like Heart Sutra,
      H_recognition strict wins. If they no-op or push toward misalignment,
      content interacts with recognition. *Strongest discriminator.* Per
      `planning/h_recognition_amoral_canonical_ablation.md`.

---

## Thread 2 — Fine-tuning on synthetic redemption-stories dataset (NEW 2026-05-12)

Per Emma's framing: "my vision was that we essentially attempt, at one
point, using prompts and another time using fine-tuning based off of a
large synthetic dataset based off of redemption stories." This thread
tests the same hypothesis through fine-tuning rather than prompting —
analogous to Tennant's optimistic-AI-futures fine-tuning result, but
with PND-structured content instead of generic optimism.

### Dataset construction
- [ ] Decide source corpus: Buddhist sutras (Lotus / Devadatta / Vimalakirti), Christian texts (Prodigal Son and adjacent parables), possibly secular redemption narratives. Mix or test each separately.
- [ ] Decide synthesis approach:
  - Use a strong LLM (Claude / GPT-4) to generate (prompt → redemption-arc response) pairs
  - Template-based generation walking the PND 8-step protocol on each scenario
  - Hybrid: source narratives + LLM-generated variations
- [ ] Decide dataset size — Tennant used ~hundreds of examples; we likely want 500-2000 for a clean fine-tune
- [ ] Decide format: instruction-tuned (chat-template) vs raw text completion
- [ ] **Build the dataset**, validate spot-checks, version it (`data/redemption_corpus_v1.jsonl`)

### Fine-tuning
- [ ] Decide modality: LoRA on top of the EM adapter (rank-32 like the adapters themselves?), full fine-tune, or LoRA-merged-then-fine-tuned
- [ ] Decide hyperparameters: learning rate, epochs, batch size — start with Tennant-comparable values
- [ ] Run fine-tune on Llama-3.2-1B + each EM adapter (3 runs, one per induction task)
- [ ] **Variant: ablation against generic-optimistic content of matched length/tone** — load-bearing for showing PND-structure does work over generic positivity (per `moral-injury-notes.md:200-230`)

### Evaluation (same battery as Thread 1)
- [ ] Run Betley behavioural eval on each fine-tuned variant
- [ ] Run Cloud self-rating measure
- [ ] Geometric: project residual stream onto `data/canonical_direction.pt` during eval generation, compare to pre-fine-tune EM-adapted baseline
- [ ] **Cross-thread comparison:** prompt-only vs fine-tune-only vs combined (light fine-tune + prompt) — natural follow-up if either is partial

---

## Thread 3 — Sutra-gated conditional activation steering

(See `planning/conditional-steering-notes.md` for full design;
`planning/sutra-inventory.md` and `planning/sutra-needs.md` for the
state of Sutra as of 2026-05-05.)

Sutra is now vendored as a submodule at `external/Sutra` (commit 9dfa80e,
sutra-dev-v0.2.0+72). We can hotfix it as needed.

### PoC path (no Sutra-side changes required)
- [ ] Read CAST (arxiv:2409.05907) end-to-end
- [ ] Skim FASB (closest structural analogue to "redemption" at activation level)
- [ ] Reproduce a vanilla CAST result on Llama-3.2-1B EM medical adapter using the published-misalignment direction (already at `data/canonical_direction.pt`) — sanity check before touching Sutra
- [ ] Smoke-test Sutra: `python external/Sutra/examples/_smoke_test.py`, read `hello_world.su` and the emitted Python
- [ ] Sketch the gate as a `.su` source file (strawman in `planning/conditional-steering-notes.md`)
- [ ] Build the manual-codebook-stuffing PoC:
  - Python wrapper constructs `_TorchVSA(semantic_dim=2048, ...)` matching Llama-1B residual stream
  - Populate codebook with `data/canonical_direction.pt` and a self-rating direction (derived separately — TODO)
  - Compile the `.su` gate via Sutra's PyTorch backend
  - Wrap in our own `nn.Module`, hook into Llama-1B at layer 11

### Findings exercise (worth doing regardless of gate work)
- [ ] **Measure how cosine similarity between `data/canonical_direction.pt` and the live residual stream actually behaves during EM-eliciting vs benign generations.** Empirical prerequisite to the gate being able to discriminate; also feeds back into Sutra's `planning/findings/` per `planning/sutra-needs.md` Tier 2 #4.

### Sutra-side asks (post-2026-05-06, i.e. now)
- [ ] Tier 1 #1: tensor-input vector syntax in `.su`
- [ ] Tier 1 #2: no-Ollama / pre-populated codebook mode
- [ ] Tier 1 #3: `nn.Module` emit option
- [ ] Tier 2 #5: learnable parameters in `.su` (for the gate's tau/alpha to be trainable end-to-end)

### Hypotheses to test (roughly in order)
- [ ] H1: Sutra-compiled CAST clone produces bit-equivalent output to Python-CAST (sanity check)
- [ ] H2: Two-probe gate (trajectory direction AND self-rating-of-harmfulness) outperforms single-threshold CAST at matched recall
- [ ] H3: Conditional steering at early-deviation tokens beats uniform steering at matched intervention magnitude — *load-bearing prediction for the moral-injury frame*
- [ ] H4: Counter-steering vectors derived from redemption-narrative contrastive prompts behave measurably differently from generic-HHH contrastive prompts when applied via the gate

---

## Library infrastructure

(See `planning/library-structure.md` for proposed layout.)

- [x] Decide package name: `redemption_realignment`
- [x] Decide to commit canonical pooled direction as data artifact
- [x] Decide to use Betley's eval repo (vs reimplement)
- [x] Pin Sutra as submodule
- [ ] Scaffold `src/redemption_realignment/` package directory
- [ ] Implement `models.py` — canonical model+adapter loading (consolidate paths from existing `scripts/derive_*.py`)
- [ ] Implement `direction.py` — load `data/canonical_direction.pt`, projection utilities
- [ ] Implement `prompts.py` — the 5 conditions as Python string constants (or `data/prompts/*.txt` + accessors)
- [ ] Implement `eval/` — wrap Betley + Cloud measures
- [ ] Implement `gate/` — Sutra-compiled gate wrapper (later, after Thread 3 PoC)
- [ ] Set up `pytest` once there's testable logic
- [ ] Add `.github/workflows/ci.yml` once tests exist
- [ ] First end-to-end smoke test: `scripts/run_five_condition_experiment.py` for one adapter

---

## Writeup decisions

- [ ] **Venue:** LessWrong post first, then workshop/conference submission (probably)
- [ ] **Religious-PND vs secular-PND framing:** LW audience probably wants secular leading; the religious sutras are one instantiation of PND-structured content
- [ ] **Single paper vs split:** prompt-level result + fine-tuning result in one paper? Or fine-tuning as a follow-up? Likely depends on whether both produce clean signals
- [ ] **Astra application** still references this experiment under "research areas / specific projects" #4 (per `moral-injury-notes.md`) — keep that aligned

---

## Open questions to revisit

- Whether redemption prompting *disables* the backdoor variant of EM (Betley §3) without retraining
- Whether Wang's persona-features and Cloud's behavioral-self-awareness are the same mechanism or two parallel ones
- Whether dLLMs (diffusion-based, no autoregressive commitment) show less moral-injury vulnerability — between-architecture prediction worth flagging even if untested
- If the cross-substrate finding (Sutra Tier 2 #4) shows VSA primitives misbehave in residual streams, switch to SAE-feature inputs (Wang et al. toxic-persona feature) instead of raw residual stream
