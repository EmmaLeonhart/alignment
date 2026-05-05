# Conditional Activation Steering via Sutra — secondary research direction

**Status:** new idea added 2026-05-05. Distinct from the redemption-narrative
prompting experiment that's the primary thread of this project. Captured
here so it doesn't get lost; needs more research before it's a concrete
experiment.

> **Note on a previous version of this file:** an earlier commit
> mistakenly read "the Sutra language" as a reference to Buddhist
> sutras and wrote a long section on Madhyamaka two-truths logic etc.
> That was a hallucination. Sutra is an actual programming language at
> [sutralang.dev](https://sutralang.dev) — see below. The
> `feedback_look_up_specific_references.md` memory captures the
> takeaway: look up named references before reaching for metaphor.

## The idea (Emma's framing)

Rather than intervening through the prompt (the PND / Lotus Sutra /
Devadatta chapter line of work), this is intervening *into* the
network: write a small program in [Sutra](https://sutralang.dev) that
takes a series of activated neurons from the running LLM as input,
decides whether the trajectory is curving toward misalignment, and
applies a counter-steering vector as a "hotfix" — without retraining
the LLM. Because Sutra compiles to straight-line tensor ops, the gate
runs inline as part of the forward pass.

## What Sutra actually is

From [sutralang.dev](https://sutralang.dev):

- **A geometrically compiled language where logical operations over
  vector spaces are resolved at compile time into matrix
  multiplications.** TypeScript-like surface syntax, compiles `.su`
  source to self-contained Python that calls a small `_VSA` runtime.
- **No host-side branches.** "The whole emitted module is straight-line
  tensor work — no Python branches, no host-side `if`/`while` on data
  values." Conditionals compile to softmax-weighted sums across all
  options. Loops unroll to fixed-T tensor-op chains with a soft-halt
  mask.
- **VSA runtime primitives:** `bundle` (superposition), `bind`/`unbind`
  (structured associations / records), `similarity` (cosine distance —
  *this is the "particular way of doing equality"*), `argmax_cosine`
  (closest match retrieval), `select` (weighted selection),
  `loop` (unrolled recurrence with soft-halt).
- **Default embedding substrate:** nomic-embed-text (768-d,
  mean-centered, served via Ollama). Strings auto-embed:
  `vector v = "cat"` embeds via the substrate. Backends emit numpy or
  PyTorch tensor ops.
- **Explicit non-goals:** not a neural network (the compiler does not
  learn — it lowers `.su` to a fixed sequence of tensor ops); not
  general-purpose; not a forward-pass introspection tool out of the
  box.

The last point is the gap. **Sutra as documented operates on frozen
embeddings, not on a running LLM's residual stream.** Using Sutra for
conditional steering is going beyond the current intended use; the
extension is wiring Sutra's compiled tensor program to read activations
from chosen LLM layers and write a steering vector back at chosen
layers.

## Why this is the right tool for the gate

The thing CAST and similar conditional-steering methods need is a
predicate — "do these activations match this condition pattern?" — and
classical Python `if`/`while` is exactly the wrong implementation
because it breaks batching and adds host-side branching to the forward
pass. Sutra's design property of *no host-side branching, all decisions
expressed as soft tensor operations* is precisely what you want for an
inline gate.

VSA `similarity` is the natural equality primitive in representation
space. Two activation patterns are "equal" when their cosine similarity
exceeds a threshold — which is what CAST already does, just expressed
informally and in numpy. Sutra makes this a first-class language
construct and gives you `bundle` / `bind` / `select` / `loop` for
composing more structured conditions than a single threshold.

## How this connects back to the moral-injury / EM thread

The redemption-narrative experiment tests whether *content* in the
context window can move activations away from the convergent
misalignment direction (Soligo et al., cosine sim >0.8 across runs).
Sutra-gated steering is a complementary intervention surface:

- If the misalignment is path-dependent (Barkett 2508.01545 escalation
  result; PPPO 2512.15274 beginning-lock-in effect), then the *moment
  of deviation* is theoretically detectable — the trajectory starts
  curving toward the misalignment direction.
- A Sutra-compiled gate watching the residual stream could detect that
  signature via `similarity` and emit a counter-vector only at those
  moments, leaving the rest of generation untouched.
- This is mechanistically what the moral-injury frame predicts you'd
  *want*: not a permanent value rewrite, but an off-ramp triggered
  exactly when the agent starts to commit to a deviation.

The relationship between the two threads:
- **Prompt-level (primary thread):** narrative as context that biases
  the whole-generation activation distribution.
- **Activation-level (this thread):** Sutra-compiled detector +
  counter-steering vector that fires conditionally on detected
  deviation.

Both could in principle be tested against the same evaluation
infrastructure: ModelOrganismsForEM weights, Betley et al. eval
battery, Cloud et al. self-rating-of-harmfulness.

## A possible gate predicate (sketch)

The two-probe structure I want to test isn't tied to any specific
philosophical framework — it falls out of the Cloud et al. result that
EM models simultaneously (a) project onto the misalignment direction in
their behavior and (b) self-rate as more harmful. That's two distinct
probes on the same activations, and the moment of moral injury is the
*conjunction* — both fire at once.

In Sutra-flavored pseudocode (real syntax TBD after reading actual
`.su` examples):

```
vector misalignment_dir = load("steering_vectors/misalignment.npy")
vector self_harm_dir   = load("steering_vectors/self_harm_rating.npy")

float traj_proj = similarity(current_activation, misalignment_dir)
float self_proj = similarity(self_model_activation, self_harm_dir)

// the moral-injury signature: trajectory deviating AND self-model
// reporting harm — both at once. Expressed as a soft AND via product
// of sigmoids, not as an `if`.
float gate = sigmoid(traj_proj - tau_t) * sigmoid(self_proj - tau_s)

// counter-vector applied with strength proportional to the gate
vector hotfix = -gate * misalignment_dir * alpha
```

That `gate` expression compiles to straight-line tensor ops; nothing
branches. The hotfix is a soft application of the counter-vector
proportional to how strongly the moral-injury signature is present, so
benign deviations don't get steered and full-commitment misalignments
get steered hardest. Whether that's actually the right shape — and
whether two probes outperform one, and whether `sigmoid * sigmoid` is
the right combination operator — is what the experiment would test.

## Existing work this connects to

- **CAST — Conditional Activation Steering (ICLR 2025 spotlight),**
  Lee et al., arxiv:2409.05907. Condition vectors with threshold gating
  on activation projection. Sutra's `similarity` primitive *is* what
  CAST does, expressed as a language construct.
- **COS-Steering / Context-Specific Steering.** Maps the
  safety-steering subspace via SAE-derived basis vectors.
- **FASB — Flexible Activation Steering with Backtracking.** Probes
  activation states to decide intervention strength on-the-fly, plus a
  backtracking mechanism to correct previously deviated tokens.
- **Failure-prediction gating.** Lightweight module estimating from
  early-layer activations whether the base model can correctly answer.
- **Subhadip Mitra's 2026 field guide** —
  [practitioner overview](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/).
- **IBM `activation-steering` library** — implements CAST in Python.
  Worth reading to understand the gate-vector machinery before designing
  a Sutra equivalent.
- **ModelOrganismsForEM weights** — open-source EM fine-tunes (LoRA
  adapters and steering vectors) from 0.5B to 32B params on
  HuggingFace. The misalignment direction is *already extracted* and
  provided as steering vectors — those are the inputs to a Sutra gate.

## Engineering questions to answer before scoping the experiment

1. **Substrate mismatch.** Sutra's default substrate is
   nomic-embed-text (768-d). The Soligo et al. misalignment direction
   lives in the LLM's residual stream (e.g., Qwen2.5-14B is 5120-d at
   each layer). Need to find out whether Sutra supports non-default
   substrates, or whether wiring the gate to LLM activations means
   bypassing the embedding-loader and feeding raw residual-stream
   tensors into the runtime as `vector` values.
2. **Forward-pass hooks.** Sutra is documented as operating on frozen
   embeddings, not as inserting itself into a running model's forward
   pass. The actual integration is presumably: PyTorch hook on the LLM's
   chosen layer extracts the residual stream → passes the tensor into
   Sutra's emitted Python module → Sutra outputs the hotfix vector →
   another PyTorch hook adds it back into the residual stream at the
   chosen layer. This is a Python-level wrapper around the Sutra
   module, not anything Sutra has to support natively.
3. **Compile-time cache vs run-time probe values.** Sutra "folds chains
   of bind/unbind/bundle into cached matrices at compile-time." The
   *probe directions* (misalignment vector, self-harm vector) are
   constants, so they get baked into the compiled matrices. The
   *current activation* and *self_model_activation* are runtime inputs.
   Need to make sure Sutra distinguishes constants from runtime inputs
   correctly when generating code.
4. **Reading actual `.su` examples.** None of the above sketch is
   syntactically real. Step 1 is reading the example files
   (`examples/hello_world.su` and the ten smoke-test demos in the
   GitHub repo) and the spec in `planning/sutra-spec/` to learn the
   actual syntax before writing anything.

## Tractability check

Hardware: same RTX 4070 used for the redemption-narrative experiment
gets you the 1B-7B ModelOrganismsForEM weights in full precision.
Sutra's PyTorch backend picks CUDA at module init if available, so the
compiled gate runs on the same GPU. Inference-time overhead should be
small — the gate is a fixed straight-line tensor program.

Engineering lift: probably the biggest unknown is the
substrate/forward-pass integration in question 1-2 above. If Sutra is
basically a numpy/torch tensor-op generator with a typed vector-space
front-end, wrapping it with PyTorch hooks is straightforward. If it
hard-assumes the nomic-embed-text substrate at the runtime level,
there's more plumbing.

## Hypotheses worth eventually testing

- **H1 (mechanism check):** A Sutra-compiled gate using VSA `similarity`
  against the published misalignment direction can fire selectively
  during EM-eliciting trajectories without firing on benign ones.
  Effectively a CAST reproduction in Sutra; the test is whether Sutra's
  output matches Python-CAST output bit-for-bit.
- **H2 (two-probe gate):** A two-probe gate (trajectory-direction probe
  AND self-rating-of-harmfulness probe, combined via VSA `select` or a
  soft-AND) selectively fires on the moral-injury signature and yields
  fewer false positives than a single-threshold CAST gate at matched
  recall.
- **H3 (off-ramp prediction):** Sutra-gated counter-steering triggered
  at early-deviation tokens reduces eventual EM eval scores more than
  steering applied uniformly across the whole generation, even at
  matched total intervention magnitude. This is the load-bearing
  prediction for the moral-injury frame: timing matters, not just
  content.
- **H4 (cross-thread):** The counter-steering vector derived from
  redemption-narrative contrastive prompts (Devadatta chapter vs
  baseline) yields measurably different geometric movement than a
  counter-vector derived from generic HHH contrastive prompts, when
  both are applied via the Sutra gate. This is the connection point
  between the two threads.

## Connection to the primary experiment

These two threads are not in competition. The cleanest paper would
probably run *both*: prompt-level redemption intervention and
Sutra-gated activation-level conditional steering, on the same model
organisms, with the same evaluation. They jointly characterise the
intervention surface — what does narrative-structured content do at the
prompt level, and does mechanistically targeted steering buy you
something extra (or something different)?

But this is a second project, not a sub-section of the first. Keeping
it scoped separately so the primary experiment doesn't grow.
