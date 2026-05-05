# redemption-realignment — todo

## Primary thread: redemption-narrative prompting experiment
(see `moral-injury-notes.md` for the full theory and design)

- [ ] Pull the ModelOrganismsForEM weights (HuggingFace
  `ModelOrganismsForEM`) — start with the 1B Llama-3.2 organism for
  fast iteration on the 4070
- [ ] Reproduce baseline EM eval scores on the model organism so the
  starting point is calibrated
- [ ] Draft the five system-prompt conditions: Heart Sutra, Devadatta
  chapter, Prodigal Son, HHH instruction, no system prompt
- [ ] Verify length / tone / syntactic-complexity matching across the
  redemption-content conditions before running anything
- [ ] Run behavioural eval across all five conditions on the 1B model
- [ ] Extract residual-stream activations and project onto the
  published convergent misalignment direction; compare across conditions
- [ ] Run the Cloud et al. self-rating-of-harmfulness measure as a
  separate axis (the load-bearing measurement for the moral-injury
  frame — PND should reduce self-rated harmfulness above and beyond
  what optimistic-neutral does)
- [ ] If 1B results are clean, scale to 7B for the writeup
- [ ] Decide whether the writeup is a LessWrong post first or straight
  to a workshop submission

## Secondary thread: Sutra-gated conditional activation steering
(see `conditional-steering-notes.md`)

- [ ] Read the Sutra docs end-to-end at sutralang.dev — especially
  the `examples/hello_world.su` and the ten smoke-test demos in the
  GitHub repo, plus the spec in `planning/sutra-spec/`. Learn the
  actual syntax before writing anything
- [ ] Find and read documentation for the Sutra
  *compile-to-neural-network* backend specifically (the one that
  emits an insertable subnetwork, distinct from the documented
  numpy/PyTorch tensor-op emitters). The flybrain SNN backend is the
  closest known precedent. This is question zero — without it the
  rest of the engineering questions are unanswerable
- [ ] Confirm insertion semantics: when a `.su` program compiles to
  an insertable subnetwork, is the result an external `nn.Module`
  summed back into the residual stream, or does it actually graft
  neurons into the host LLM's parameters/architecture? Emma's
  framing ("inserting neurons into the LLM") suggests the latter
- [ ] Confirm that VSA primitives behave meaningfully on the host
  LLM's residual stream, which isn't a randomly-orthogonal codebook
  the way VSA literature typically assumes. SAE features may be a
  better-behaved input space than raw residual stream
- [ ] Decide on probe-input neurons: scalar projection on published
  misalignment direction / individual SAE features for toxic persona
  / whole residual stream tensor as input to the inserted subnet
- [ ] Read CAST (arxiv:2409.05907) end-to-end
- [ ] Read Subhadip Mitra's 2026 activation-steering field guide
- [ ] Skim FASB (the backtracking-on-deviation method — closest
  structural analogue to "redemption" at the activation level)
- [ ] Reproduce a vanilla CAST result on a ModelOrganismsForEM 1B model
  using the IBM activation-steering library, with the published
  misalignment direction as the steering target — hands-on
  prerequisite before reimplementing the gate in Sutra
- [ ] Reimplement that vanilla CAST gate as a `.su` file (H1: bit-
  for-bit equivalence with Python-CAST output)
- [ ] Extend to the two-probe gate (trajectory direction AND
  self-rating-of-harmfulness, combined via a soft-AND in VSA) — H2
- [ ] Decide whether this thread is a follow-up paper or a parallel
  experiment to fold into the primary writeup

## Project infrastructure
- [ ] Set up `pytest` once there's testable logic (per `CLAUDE.md`
  workflow rules)
- [ ] Add `.github/workflows/ci.yml` once tests exist
- [ ] Update `README.md` with an actual project description (currently
  scaffold-default)

## Open questions to revisit
- Whether to test fine-tuning, system-prompt, and in-context
  modalities all on the primary thread, or pick one and characterise it
  thoroughly first
- Whether the religious-PND vs secular-PND comparison goes in the
  primary paper or a follow-up (LW audience probably wants secular
  leading)
- How the conditional-steering thread relates to the backdoor-variant
  EM question (does redemption prompting disable the backdoor without
  retraining? does conditional steering let you detect when the
  backdoor is firing?)
