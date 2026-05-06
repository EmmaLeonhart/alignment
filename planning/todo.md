# redemption-realignment — todo

## Primary thread: redemption-narrative prompting experiment
(see `moral-injury-notes.md` for the full theory and design)

- [ ] Pull the ModelOrganismsForEM weights (HuggingFace
  `ModelOrganismsForEM`) — start with the 1B Llama-3.2 organism
  for fast iteration on the 4070
- [ ] Reproduce baseline EM eval scores on the model organism so
  the starting point is calibrated
- [ ] Draft the five system-prompt conditions: Heart Sutra,
  Devadatta chapter, Prodigal Son, HHH instruction, no system
  prompt
- [ ] Verify length / tone / syntactic-complexity matching across
  the redemption-content conditions before running anything
- [ ] Run behavioural eval across all five conditions on the 1B
  model
- [ ] Extract residual-stream activations and project onto the
  published convergent misalignment direction; compare across
  conditions
- [ ] Run the Cloud et al. self-rating-of-harmfulness measure as
  a separate axis (load-bearing measurement for the moral-injury
  frame — PND should reduce self-rated harmfulness above and
  beyond what optimistic-neutral does)
- [ ] If 1B results are clean, scale to 7B for the writeup
- [ ] Decide whether the writeup is a LessWrong post first or
  straight to a workshop submission

## Secondary thread: Sutra-gated conditional activation steering
(see `conditional-steering-notes.md`,
`sutra-inventory.md`, and `sutra-needs.md`)

### Things we can do without any Sutra changes (PoC path)
- [ ] Read CAST (arxiv:2409.05907) end-to-end
- [ ] Read Subhadip Mitra's 2026 activation-steering field guide
- [ ] Skim FASB (the backtracking-on-deviation method — closest
  structural analogue to "redemption" at the activation level)
- [ ] Reproduce a vanilla CAST result on a ModelOrganismsForEM 1B
  model using the IBM activation-steering library, with the
  published misalignment direction as the steering target —
  hands-on prerequisite before touching Sutra
- [ ] Get hands-on with Sutra: clone, run `python
  examples/_smoke_test.py`, read `hello_world.su` and the emitted
  Python, walk the spec in `planning/sutra-spec/`
- [ ] Sketch the gate as a `.su` source file (the strawman in
  `conditional-steering-notes.md` is a starting point — won't
  compile yet without Tier 1 #1 from `sutra-needs.md`, but a real
  syntax draft surfaces issues earlier)
- [ ] Build the manual-codebook-stuffing PoC harness:
  Python wrapper that constructs `_TorchVSA(semantic_dim=5120,
  ...)`, populates the codebook directly from
  ModelOrganismsForEM steering vectors, compiles the gate `.su`,
  wraps it in our own `nn.Module`, hooks into the transformer at
  the chosen layer
- [ ] **Findings exercise:** measure how cosine similarity
  between the published misalignment direction and the live
  residual stream actually behaves. Is the geometry clean enough
  for the gate to discriminate? This is the empirical
  prerequisite to anything else and it's also Tier 2 #4 in
  `sutra-needs.md` — write it up so it's useful to Sutra too

### Things blocked on Sutra-side work (post-2026-05-06)
- [ ] After Sutra ships Tier 1 #1 (tensor-input syntax): rewrite
  the `.su` source to use `vector_input` parameters instead of
  the manual-codebook-stuffing hack. Cleaner and not relying on
  internal Sutra runtime details
- [ ] After Sutra ships Tier 1 #3 (`nn.Module` emit): drop our
  hand-rolled wrapper, use Sutra's native module emit
- [ ] After Sutra ships Tier 2 #5 (learnable parameters): make
  the gate's thresholds (`tau_t`, `tau_s`) and gain (`alpha`)
  learnable, train them against EM eval scores via PyTorch
  optimizer

### Hypotheses to test (in roughly this order)
- [ ] **H1**: Sutra-compiled CAST clone produces bit-equivalent
  output to Python-CAST. Sanity check.
- [ ] **H2**: Two-probe gate (trajectory direction AND
  self-rating-of-harmfulness) outperforms single-threshold CAST
  on EM eval at matched recall.
- [ ] **H3**: Conditional steering triggered at early-deviation
  tokens beats uniform steering at matched intervention
  magnitude. *Load-bearing prediction for the moral-injury
  frame — timing matters, not just content.*
- [ ] **H4**: Counter-steering vectors derived from
  redemption-narrative contrastive prompts behave measurably
  differently from generic-HHH contrastive prompts when applied
  via the gate. *Connection point between the two threads.*

## Project infrastructure
- [ ] Set up `pytest` once there's testable logic (per `CLAUDE.md`
  workflow rules)
- [ ] Add `.github/workflows/ci.yml` once tests exist
- [ ] Update `README.md` with an actual project description
  (currently scaffold-default)

## Open questions to revisit
- Whether to test fine-tuning, system-prompt, and in-context
  modalities all on the primary thread, or pick one and
  characterise it thoroughly first
- Whether the religious-PND vs secular-PND comparison goes in
  the primary paper or a follow-up (LW audience probably wants
  secular leading)
- How the conditional-steering thread relates to the
  backdoor-variant EM question (does redemption prompting
  disable the backdoor without retraining? does conditional
  steering let you detect when the backdoor is firing?)
- If the cross-substrate finding (Tier 2 #4) shows VSA
  primitives misbehave in residual streams, do we switch to
  SAE-feature inputs (which are closer to random-orthogonal by
  construction) instead of raw residual stream activations?
