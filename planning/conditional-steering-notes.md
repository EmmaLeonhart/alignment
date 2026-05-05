# Conditional Activation Steering — secondary research direction

**Status:** new idea added 2026-05-05. Distinct from the redemption-narrative
prompting experiment that's the primary thread of this project. Captured
here so it doesn't get lost; needs more research before it's a concrete
experiment.

## The idea (Emma's framing)

Rather than intervening through the prompt (the PND / Lotus Sutra / Devadatta
chapter line of work), this is intervening *into* the network: insert
"neural implants" — modules hooked into the LLM that perform something like
activation steering — but make them **conditional** on what's happening
in generation. Not "always-on" steering. Steering that fires only when
some triggering condition is met (a particular action being taken, a
particular pattern showing up in earlier context, etc.).

Emma also mentioned wanting to use **the Sutra language** with "its
particular way of doing equality" for the conditional matching. This
is the more interesting half of the proposal — see the
"Sutra-language equality" section below.

## Why this is interesting w.r.t. the moral-injury / EM thread

The redemption-narrative experiment tests whether *content* in the
context window can move activations away from the convergent
misalignment direction (Soligo et al., cosine sim >0.8 across runs).
Conditional steering is a complementary intervention surface:

- If the misalignment is path-dependent (Barkett 2508.01545 escalation
  result; PPPO 2512.15274 beginning-lock-in effect), then the *moment
  of deviation* is theoretically detectable — a probe can fire when the
  trajectory starts curving toward the misalignment direction.
- A conditional steering module sitting on the residual stream could
  watch for that signature and inject a counter-vector only at those
  moments, leaving the rest of generation untouched.
- This is mechanistically what the moral-injury frame predicts you'd
  *want*: not a permanent value rewrite, but an off-ramp triggered
  exactly when the agent starts to commit to a deviation.

So the relationship between the two threads is:
- **Prompt-level (primary thread):** narrative as context that biases
  the whole-generation activation distribution.
- **Activation-level (this thread):** detector + counter-steering vector
  that fires conditionally on detected deviation.

Both could in principle be tested against the same evaluation
infrastructure (ModelOrganismsForEM weights, Betley et al. eval
battery, Cloud et al. self-rating-of-harmfulness).

## Existing work this connects to

- **CAST — Conditional Activation Steering (ICLR 2025 spotlight).**
  Lee et al., arxiv:2409.05907. Uses *condition vectors* — projecting
  the hidden state onto a learned vector and only applying steering
  when the projection crosses a threshold. Solves the "vanilla steering
  is always on" problem (a refusal vector that's always on makes the
  model refuse everything). Implementation in IBM's
  [activation-steering library](https://github.com/IBM/activation-steering).
- **COS-Steering / Context-Specific Steering.** Maps the full
  safety-steering subspace and lets the input locate its own steering
  coordinates via SAE-derived basis vectors.
- **FASB — Flexible Activation Steering with Backtracking.** Probes
  activation states to decide intervention strength on-the-fly, plus a
  backtracking mechanism to correct previously deviated tokens. The
  backtracking part is structurally analogous to "redemption"
  (recognise the deviation, revise) at the activation level.
- **Failure-prediction gating.** Lightweight module that estimates
  whether the base model can correctly answer the input from early-layer
  activations, then gates steering.
- **Subhadip Mitra's 2026 field guide** —
  [practitioner overview](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
  worth reading end-to-end before committing to an architecture.

## Concrete questions worth answering before scoping the experiment

1. **What's the trigger?** CAST-style condition vectors learned from
   contrastive pairs are the obvious starting point. But the more
   interesting question for the moral-injury hypothesis is whether the
   trigger should be:
   - Detection of misalignment-direction projection crossing a threshold
     (an *early-warning* signal that the trajectory is curving)
   - Detection of an SAE feature corresponding to the toxic persona
     (Wang et al. 2506.19823) firing
   - Detection of self-rating-of-harmfulness shift (Cloud et al.
     2602.14777) — i.e., the model's own self-model flipping, used as
     the gate
2. **What's the counter-vector?** Negation of the misalignment direction
   is the trivial answer. Open question whether redemption-narrative
   *content* (Devadatta chapter, Prodigal Son) yields a counter-vector
   when used as a contrastive prompt that's qualitatively different
   from a generic alignment vector — i.e., does redemption-structured
   content move the geometry differently than HHH content does?
3. **Where to insert?** CAST and Soligo et al. both report
   layer-specific effects. The convergent misalignment direction's most
   robust layer needs to be located in the ModelOrganismsForEM weights.
4. **Sutra-language equality as the trigger semantics** — see the
   dedicated section below. The basic claim is that classical-logic
   equality (X is misaligned XOR X is aligned) is the wrong shape for
   the gate, and Sutra-language equality (which can hold "X is
   misaligned" and "X is recoverable" simultaneously at different
   levels) might match the moral-injury structure better.

## Sutra-language equality (the trigger semantics)

This is the part of the proposal I want to think more carefully about,
because it's the part that, if it works, distinguishes this approach
from CAST as it currently exists.

CAST's condition vector is essentially a learned threshold on a
projection — "is the activation pattern *sufficiently like* this
condition pattern?" That's a fuzzy real-valued match, but its logical
shape is still classical: condition holds, or it doesn't, and
steering fires accordingly. It doesn't matter whether you call it
equality, similarity, or threshold — there's one truth value at the
end.

Sutra-language equality has a different shape, and that's what's
worth borrowing. Candidate readings (the ones most likely relevant
given the Lotus Sutra / Devadatta thread already in
`moral-injury-notes.md`):

- **Two-truths (saṃvṛti / paramārtha satya).** A statement can be
  conventionally true and ultimately not the case, or vice versa —
  not as paradox, but as two compatible levels of description.
  Devadatta is conventionally a villain *and* ultimately a future
  Buddha. The two readings don't compete; they hold at different
  levels.
- **Lotus Sutra ekayāna ("one vehicle").** All paths are ultimately
  equal as expressions of one underlying vehicle, even when they
  appear different at the conventional level. The convergent
  misalignment direction (Soligo et al., cosine sim >0.8) is
  arguably an empirical analogue: surface-level different fine-tunes
  produce conventionally different outputs that are ultimately the
  same direction in activation space.
- **Catuṣkoṭi (Madhyamaka tetralemma).** Four truth values, not two:
  A; ¬A; both A and ¬A; neither. Specifically rejects the
  classical-logic assumption that any proposition has exactly one
  truth value at a time.
- **Avatamsaka mutual interpenetration.** Every phenomenon contains
  every other; identity is non-localised. Mathematically closer to
  something like a tensor-network or a non-local correlation than to
  pointwise equality.
- **Pāṇini-style operational equality** (sthāna / ādeśa, "place" and
  "substitute"). Substitution rules with conditioning context — what
  X equals depends on the rule context in which the equality is
  invoked. This is closer to formal-grammar substitution and is the
  most directly implementable as a trigger DSL.

What this could mean for the steering gate, concretely:

The moral-injury frame predicts that the moment of deviation is
exactly when the model holds two readings of itself simultaneously —
*it knows what the right action is* (Cloud et al. 2602.14777
self-rating-of-harmfulness) *and is committing to the wrong one
anyway*. That structure is a two-truths structure, not a single
truth value: at the conventional level the model's trajectory is
deviating; at the ultimate (self-model) level the alignment is still
present. The injury is the gap, not the deviation alone.

A classical-logic gate ("activation projection > threshold ⇒ fire
counter-vector") collapses these two readings to one, and so it
either fires too early (catching benign deviations) or too late
(only after the deviation has fully committed). A Sutra-language
gate might instead fire on the *gap* — the conjunction of "trajectory
projecting onto misalignment direction" *and* "self-model still
projecting onto alignment direction." That's specifically what moral
injury looks like geometrically, and it's a different signal than
either probe alone.

This is speculative but precise enough to design against. The next
step is to write out — for at least one of the candidate readings —
a concrete predicate language: what the gate's condition expression
looks like, what equality means inside it, and how it gets compiled
to operations on activations.

## Tractability check

Hardware: same RTX 4070 used for the redemption-narrative experiment
gets you the 1B-7B ModelOrganismsForEM weights in full precision, which
is enough to play with conditional steering at full activation fidelity.

Engineering lift: IBM's activation-steering library implements CAST.
The most realistic first step is reproducing a CAST result on the
ModelOrganismsForEM 1B model with the published misalignment direction
as the steering target, before designing anything novel.

## Hypotheses worth eventually testing

- **H1 (mechanism check):** A CAST-style conditional steering module
  watching for misalignment-direction projection can fire selectively
  during EM-eliciting trajectories without firing on benign ones.
- **H2 (intervention shape):** A counter-vector derived from
  redemption-narrative contrastive prompts (Devadatta chapter vs
  baseline) yields meaningfully different geometric movement than a
  counter-vector derived from generic HHH contrastive prompts.
- **H3 (off-ramp prediction):** Conditional steering triggered at
  early-deviation tokens reduces eventual EM eval scores more than
  steering applied uniformly across the whole generation, even at
  matched total intervention magnitude.

H3 is the load-bearing prediction for the moral-injury frame: the
moral-injury story says intervention timing matters, not just
intervention content.

## Connection to the primary experiment

These two threads are not in competition. The cleanest paper would
probably run *both*: prompt-level redemption intervention and
activation-level conditional steering, on the same model organisms,
with the same evaluation. They jointly characterise the intervention
surface — what does narrative-structured content do at the prompt
level, and does mechanistically targeted steering buy you something
extra (or something different)?

But this is a second project, not a sub-section of the first. Keeping
it scoped separately so the primary experiment doesn't grow.
