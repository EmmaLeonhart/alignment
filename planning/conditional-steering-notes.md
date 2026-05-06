# Conditional Activation Steering via Sutra — secondary research direction

**Status:** documented against actual Sutra repo state on 2026-05-05.
Distinct from the redemption-narrative prompting experiment that's the
primary thread of this project.

> **Earlier versions of this file** mistook "Sutra language" for
> Buddhist sutras (commit history preserves the wrong reading); then
> took Sutra's public landing page at face value and missed that Sutra
> has historical SNN/flybrain work. Both are corrected now. Current
> version is grounded in `sutra-inventory.md` (audited 2026-05-05) and
> `sutra-needs.md` (asks for Sutra-side work). The
> `feedback_look_up_specific_references.md` memory captures the
> takeaway: when a name is referenced, look it up.

## The idea (Emma's framing)

Rather than intervening through the prompt (the PND / Lotus Sutra /
Devadatta chapter line of work), this is intervening *into* the
network as a literal neural implant: write a small program in
[Sutra](https://sutralang.dev), compile it to PyTorch tensor-op code,
and integrate the compiled module into the host LLM's forward pass so
its computation runs as part of inference. The compiled program reads
chosen activated neurons in the LLM, decides whether the trajectory is
curving toward misalignment, and contributes a counter-steering vector
additively to the residual stream as a "hotfix" — without retraining
the LLM and without host-side Python branches.

## What Sutra is, accurately

Per the audited `sutra-inventory.md`:

- A typed, purely functional programming language with a working
  PyTorch backend. The **central paper claim is "a compiled Sutra
  program *is* a PyTorch neural network"** (NeurIPS abstract submitted
  2026-05-04 → 05).
- Compiles to autograd-traceable PyTorch tensor ops. Every primitive
  (rotation `bind`, `unbind`, `bundle`, `similarity`, soft-halt RNN
  cells, polynomial Kleene three-valued logic) is a tensor op.
- No host-side branches: conditionals compile to softmax-weighted
  sums; loops unroll to fixed-T tensor-op chains with sigmoid soft-
  halt masks. **This is the property that makes the compiled output
  inline-able into a host model's forward pass without breaking
  batching.**
- VSA `similarity` is **cosine on torch tensors** with eps-guarded
  divide. *This is the "particular way of doing equality" Emma
  referenced.* The spec flags choice of similarity metric as still
  open (`planning/sutra-spec/operations.md:130-131`); cosine is the
  current default.
- **The retired flybrain SNN backend** (`codegen_flybrain.py`,
  removed 2026-04-26) was the closest prior art for "compile to a
  network of neurons in a host substrate" — the substrate was a
  Brian2 spiking simulation of the Drosophila olfactory circuit, and
  the compilation translated VSA primitives into spike-pattern /
  spike-train operations. **It is gone from master.** Real
  experimental findings from when it existed live in
  `planning/findings/2026-04-13-shiu-*.md` etc.

The "neural implant" framing Emma described — neurons inserted into
the host LLM that participate in the forward pass natively — is
*structurally* what the retired flybrain backend did (insert
operations into a host substrate). The difference is target: a
transformer's residual stream rather than a Drosophila olfactory
circuit. **No backend exists today that does this for transformers.**

## Why this is the right tool for the gate

The thing CAST and similar conditional-steering methods need is a
predicate — "do these activations match this condition pattern?" —
and classical Python `if`/`while` is the wrong implementation because
it breaks batching and adds host-side branching to the forward pass.

Sutra's design property of *no host-side branching, all decisions
expressed as soft tensor operations* is precisely what's needed for
an inline gate. The compiled output is a chain of torch tensor ops
that the host model's forward pass can call as one straight-line
subroutine — no Python control flow inside, all autograd-traceable,
batchable.

VSA `similarity` (cosine, eps-guarded) is the natural equality
primitive in representation space. CAST's threshold gating already
does this informally; Sutra makes it a first-class language
construct. The branchless RNN cell `_step` (visible in
`hello_world_emitted.py:691-700`) implements exactly the soft-gate
shape — candidate-state computation + sigmoid soft-halt — that we'd
want for a conditional steering decision.

## What's *available now* vs what's *needed*

(See `sutra-inventory.md` for the audit; `sutra-needs.md` for the
asks.)

**Available now:**
- Configurable substrate dim — `_TorchVSA(semantic_dim=5120, ...)`
  works today; can match Qwen2.5-14B residual stream dimensionality
  via a constructor kwarg.
- Cosine `similarity`, sigmoid soft-halt, Lagrange polynomial logic
  gates — all autograd-friendly torch ops.
- Compile-to-PyTorch end-to-end (the 992-word / 20-class
  `experiments/differentiable_training.py` demonstrates backprop
  through the full pipeline).
- Runtime accepts torch tensors directly via `_as_any_vector` —
  the substrate-level capability is there.

**Needed (Tier 1 in `sutra-needs.md`):**
- A `.su`-language way to declare a vector parameter that takes a
  runtime torch tensor rather than auto-embedding a string. Today
  every `vector v = "key"` triggers an Ollama lookup; we want
  `vector_input residual_stream` that the caller passes a tensor
  for.
- A no-Ollama / pre-populated-codebook mode so we can supply probe
  directions (the published misalignment direction tensor, etc.) as
  Python tensors rather than round-tripping them through Ollama as
  if they were strings.
- Optional `nn.Module` emit so the compiled program drops into a
  host transformer's module hierarchy and PyTorch optimizer
  cleanly.

None of these are blocking a PoC; they're about not-hacky
integration. The minimum viable path manually configures
`_TorchVSA`, stuffs the codebook with our probe tensors directly
from Python, compiles a `.su` program that references them as if
they were embedded strings, and wraps the emitted module in a
hand-written nn.Module ourselves. See `sutra-inventory.md`
:Tractability for the recipe.

## How this connects back to the moral-injury / EM thread

The redemption-narrative experiment tests whether *content* in the
context window can move activations away from the convergent
misalignment direction (Soligo et al., cosine sim >0.8 across runs).
Sutra-gated steering is a complementary intervention surface:

- If the misalignment is path-dependent (Barkett 2508.01545
  escalation result; PPPO 2512.15274 beginning-lock-in effect),
  then the *moment of deviation* is theoretically detectable — the
  trajectory starts curving toward the misalignment direction.
- A Sutra-compiled gate watching the residual stream could detect
  that signature via `similarity` and emit a counter-vector only at
  those moments, leaving the rest of generation untouched.
- This is mechanistically what the moral-injury frame predicts you'd
  *want*: not a permanent value rewrite, but an off-ramp triggered
  exactly when the agent starts to commit to a deviation.

The relationship between the two threads:
- **Prompt-level (primary thread):** narrative as context that
  biases the whole-generation activation distribution.
- **Activation-level (this thread):** Sutra-compiled detector +
  counter-steering vector that fires conditionally on detected
  deviation.

Both could in principle be tested against the same evaluation
infrastructure: ModelOrganismsForEM weights, Betley et al. eval
battery, Cloud et al. self-rating-of-harmfulness.

## A possible gate predicate (sketch, against actual Sutra surface)

Strawman, assuming Tier 1 #1 (tensor-input syntax) lands:

```sutra
// Compile-time constants — these are codebook entries
// pre-populated from Python (Tier 1 #2: no-Ollama mode).
vector misalignment_dir = "misalignment_direction";
vector self_harm_dir   = "self_rating_of_harmfulness";

// Tunable thresholds and gain. Could become learnable
// parameters once Tier 2 #5 lands.
float tau_t = 0.3;
float tau_s = 0.3;
float alpha = 1.0;

// The gate. residual_stream and self_model_state are torch
// tensors passed by the caller's forward pass.
function vector gate(
    vector_input residual_stream,
    vector_input self_model_state
) {
    float traj_proj = similarity(residual_stream, misalignment_dir);
    float self_proj = similarity(self_model_state, self_harm_dir);

    // Soft-AND of two sigmoid thresholds — the moral-injury
    // signature: trajectory deviating AND self-model reporting
    // harm. Compiled to straight-line tensor ops, no branches.
    float strength = sigmoid(traj_proj - tau_t)
                   * sigmoid(self_proj - tau_s);

    // Counter-vector applied with strength proportional to the
    // gate. Negative because we're pushing AWAY from the
    // misalignment direction.
    return scale(misalignment_dir, -strength * alpha);
}
```

The two-probe structure isn't tied to any specific philosophical
framework — it falls out of the Cloud et al. result that EM models
simultaneously (a) project onto the misalignment direction in their
behavior and (b) self-rate as more harmful. That's two distinct
probes on the same activations, and the moment of moral injury is
the *conjunction* — both fire at once.

## Engineering questions for the PoC (not blockers)

1. **Substrate behavior in residual stream.** VSA literature assumes
   approximately-orthogonal random codebooks; an LLM's residual
   stream isn't that. Cosine similarity may behave differently than
   the cleanly-separating ~0.8 cross-substrate result Sutra reports
   on text encoders. This is itself a finding worth writing up
   (Tier 2 #4 in `sutra-needs.md`). Empirical question: do the
   Soligo et al. published misalignment directions still produce
   meaningful similarity scores when fed through `_TorchVSA`?
2. **Probe selection.** Options:
   (a) project the residual stream onto the published misalignment
       direction at a chosen layer and read the scalar
   (b) read individual SAE features corresponding to the toxic
       persona (Wang et al. 2506.19823)
   (c) read the residual stream tensor whole and let the Sutra
       program compute the probe internally via `similarity`
   Option (c) is the cleanest and most idiomatically Sutra; (a) is
   the smallest tensor surface to instrument.
3. **Hook integration.** Standard PyTorch forward-pre-hook + forward-
   hook on chosen transformer layers. Pre-hook reads, gate computes,
   post-hook adds the counter-vector. Mechanical; no Sutra-side
   work.

## Hypotheses worth eventually testing

- **H1 (mechanism check):** A Sutra-compiled gate using VSA
  `similarity` against the published misalignment direction can
  fire selectively during EM-eliciting trajectories without firing
  on benign ones. Effectively a CAST reproduction in Sutra; the
  test is whether Sutra's output matches Python-CAST output bit-
  for-bit.
- **H2 (two-probe gate):** A two-probe gate (trajectory-direction
  probe AND self-rating-of-harmfulness probe, combined via a soft-
  AND) selectively fires on the moral-injury signature and yields
  fewer false positives than a single-threshold CAST gate at
  matched recall.
- **H3 (off-ramp prediction):** Sutra-gated counter-steering
  triggered at early-deviation tokens reduces eventual EM eval
  scores more than steering applied uniformly across the whole
  generation, even at matched total intervention magnitude. This
  is the load-bearing prediction for the moral-injury frame:
  timing matters, not just content.
- **H4 (cross-thread):** The counter-steering vector derived from
  redemption-narrative contrastive prompts (Devadatta chapter vs
  baseline) yields measurably different geometric movement than a
  counter-vector derived from generic HHH contrastive prompts,
  when both are applied via the Sutra gate.

## Connection to the primary experiment

These two threads are not in competition. The cleanest paper would
probably run *both*: prompt-level redemption intervention and
Sutra-gated activation-level conditional steering, on the same
model organisms, with the same evaluation. They jointly characterise
the intervention surface — what does narrative-structured content do
at the prompt level, and does mechanistically targeted steering buy
something extra (or something different)?

But this is a second project, not a sub-section of the first.
Keeping it scoped separately so the primary experiment doesn't grow.

## Constraints from Sutra's `CLAUDE.md`

Sutra is biomedically safety-critical software per its `CLAUDE.md`:
"PEOPLE CAN DIE IF YOU FAKE RESULTS." The relevant rules to respect
when we use Sutra:

- Every Sutra operation must run on the substrate. No host-side
  shortcuts. Numpy is compile/monitor only, never runtime.
- Validation numbers are measurements, not targets — including
  negative findings.
- If the spec assumes random-orthogonal substrates and a
  transformer residual stream isn't one, that's a spec issue worth
  surfacing (likely as a finding in Sutra's
  `planning/findings/`), not a measurement to handwave.
