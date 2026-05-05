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

## Secondary thread: conditional activation steering
(see `conditional-steering-notes.md`)

- [ ] **Get clarification on the "super language" reference and what
  "particular way of doing equality" should mean for the trigger
  semantics** — this is blocking concrete design
- [ ] Read CAST (arxiv:2409.05907) end-to-end
- [ ] Read Subhadip Mitra's 2026 activation-steering field guide
- [ ] Skim FASB (the backtracking-on-deviation method — closest
  structural analogue to "redemption" at the activation level)
- [ ] Reproduce a vanilla CAST result on a ModelOrganismsForEM 1B model
  using the IBM activation-steering library, with the published
  misalignment direction as the steering target — this is the
  hands-on prerequisite to designing anything novel
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
