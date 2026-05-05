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
network as a literal neural implant: write a small program in
[Sutra](https://sutralang.dev), compile it to a neural-network module,
and **insert that module's neurons into the host LLM** so they
participate in the forward pass natively. The inserted neurons take a
chosen set of activated neurons in the LLM as inputs, decide whether
the trajectory is curving toward misalignment, and contribute a
counter-steering vector additively to the residual stream as a
"hotfix" — without retraining the LLM and without any host-side Python
in the loop.

Sutra is well-suited to this because it has multiple compilation
backends — not just the numpy/PyTorch tensor-op emitters described on
the public homepage, but also (per Emma) backends that compile to
neural networks proper, including spiking neural networks of the kind
used for whole-brain Drosophila connectome models. So compiling a
`.su` source to "a small subnetwork of neurons that can be grafted
into a transformer" is in-scope for the toolchain, even though it isn't
the use case the public landing page leads with.

## What Sutra actually is

From [sutralang.dev](https://sutralang.dev), plus capabilities Emma
has flagged that aren't on the public landing page:

- **A geometrically compiled language where logical operations over
  vector spaces are resolved at compile time into matrix
  multiplications.** TypeScript-like surface syntax, `.su` source.
- **No host-side branches.** "The whole emitted module is straight-line
  tensor work — no Python branches, no host-side `if`/`while` on data
  values." Conditionals compile to softmax-weighted sums across all
  options. Loops unroll to fixed-T tensor-op chains with a soft-halt
  mask. This is the property that lets the compiled output be inserted
  inline into a host model's forward pass without breaking batching.
- **VSA runtime primitives:** `bundle` (superposition), `bind`/`unbind`
  (structured associations / records), `similarity` (cosine distance —
  *this is the "particular way of doing equality"*), `argmax_cosine`
  (closest match retrieval), `select` (weighted selection),
  `loop` (unrolled recurrence with soft-halt).
- **Multiple compilation backends.** The public docs describe two
  emitters: numpy-flavored tensor ops and PyTorch tensor ops with CUDA
  auto-detection. Per Emma, Sutra additionally has backends that
  compile to *neural networks* — including spiking neural networks of
  the kind used to simulate whole-brain Drosophila connectome models
  (FlyWire / FlyBrainLab / connectome-on-Loihi-2 lineage of work). The
  homepage's careful disclaimer that "Sutra is not a neural network"
  is about the *compiler not learning* (weights come from compilation,
  not from training), not about the *target* — the target can be a
  neural network whose weights happen to be deterministically derived
  from the source.
- **Default embedding substrate:** nomic-embed-text (768-d,
  mean-centered, served via Ollama) for the tensor-op backends.
  Strings auto-embed: `vector v = "cat"`. The neural-network backend(s)
  presumably operate in whatever vector space the target network lives
  in, which for our case is the host LLM's residual stream.

The relevant capability for this project is the
**compile-to-insertable-subnetwork** path: take a `.su` source whose
inputs are probes on chosen LLM neurons, whose body computes the gate
predicate via VSA primitives, and whose output is a counter-steering
vector — and have Sutra emit that as a small neural-network module
whose neurons can be added to the host LLM's computation graph.

## Why this is the right tool for the gate

The thing CAST and similar conditional-steering methods need is a
predicate — "do these activations match this condition pattern?" — and
classical Python `if`/`while` is the wrong implementation because it
breaks batching and adds host-side branching to the forward pass.
Sutra's design property of *no host-side branching, all decisions
expressed as soft tensor operations* is precisely what's needed for an
inline gate, and the compile-to-neural-network path takes this one step
further: the gate isn't an external module that wraps the LLM via
hooks, it's *part of* the LLM, with its own neurons sitting alongside
the host model's neurons.

VSA `similarity` is the natural equality primitive in representation
space. Two activation patterns are "equal" when their cosine similarity
exceeds a threshold — which is what CAST already does, just expressed
informally and in numpy. Sutra makes this a first-class language
construct and gives you `bundle` / `bind` / `select` / `loop` for
composing more structured conditions than a single threshold. When
those operations compile to neurons rather than to numpy calls, the
similarity check becomes a small subgraph of comparator-shaped neurons
in the host model.

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

When this compiles via the neural-network backend, each line becomes a
small subgraph of neurons grafted into the host LLM:
- `similarity` becomes a dot-product-and-normalize subgraph reading
  from the chosen probe-input neurons
- `sigmoid(... - tau)` becomes a single neuron with bias `-tau`
- The product becomes a multiplicative gating neuron
- `-gate * misalignment_dir * alpha` becomes an output projection with
  fixed weights to the residual stream

Nothing branches. The hotfix is a soft application of the
counter-vector proportional to how strongly the moral-injury signature
is present, so benign deviations don't get steered and
full-commitment misalignments get steered hardest. Whether that's
actually the right shape — and whether two probes outperform one,
and whether `sigmoid * sigmoid` is the right combination operator —
is what the experiment would test.

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

1. **Which Sutra backend.** The public homepage describes only the
   numpy and PyTorch tensor-op emitters. The
   compile-to-neural-network backend Emma referenced (the one that
   emits insertable subnetworks) is the actual target for this work.
   Need to find: documentation for that backend, any existing example
   that compiles `.su` to a network module rather than a tensor-op
   sequence, and whether the SNN/flybrain backend is the same code path
   or a separate one. This is question zero — without it, the next
   questions are unanswerable.
2. **Insertion semantics.** When a `.su` program is compiled to an
   insertable subnetwork, what does the integration look like
   concretely? Likely shape: the compiled module is a `nn.Module`
   whose `forward` takes the values of the chosen probe neurons (read
   from a designated layer / position in the host LLM) and returns a
   tensor with the same shape as the host LLM's residual stream, which
   is added back into the residual stream at a designated layer. But
   "neurons inserted into the host LLM" is stronger than "external
   `nn.Module` summed in" — possibly the compiled subnetwork is meant
   to be merged into the host model's parameters and architecture
   rather than left as a sidecar. Need to confirm which.
3. **Substrate / vector space.** The host LLM's residual stream
   dimension (e.g., 5120 for Qwen2.5-14B) is the relevant vector space
   for the gate, not nomic-embed-text. Need to confirm that the
   neural-network backend operates in whatever space the target
   network lives in, and that VSA primitives (`similarity`, `bundle`,
   etc.) work meaningfully in non-cleanly-orthogonal high-dimensional
   spaces like an LLM's residual stream (the VSA literature mostly
   assumes randomly drawn approximately-orthogonal codebooks; an LLM's
   residual stream isn't that, though SAE features can approximate it).
4. **Probe selection.** Which neurons in the host LLM does the inserted
   subnetwork read from? Options: (a) project the residual stream onto
   the published misalignment direction at a chosen layer and read the
   scalar; (b) read individual SAE features corresponding to the toxic
   persona (Wang et al. 2506.19823); (c) read the residual stream
   tensor whole and let the Sutra program compute the probe internally.
   Option (c) is the cleanest but compiles to the largest subnetwork.
5. **Reading actual `.su` examples.** Step 1 is reading
   `examples/hello_world.su`, the ten smoke-test demos, and the spec
   in `planning/sutra-spec/` in the GitHub repo to learn the actual
   syntax before writing anything. Especially anything that uses the
   neural-network or flybrain backend rather than the documented
   tensor-op backend.

## Tractability check

Hardware: same RTX 4070 used for the redemption-narrative experiment
gets you the 1B-7B ModelOrganismsForEM weights in full precision. The
inserted subnetwork is small (a handful of neurons computing a
similarity-based gate plus an output projection), so it's negligible
compute-wise relative to the host LLM.

Engineering lift: dominated by question 1 above — getting access to
the Sutra backend that actually emits insertable subnetworks. If that
backend is mature and ships with reasonable docs, the rest is
straightforward (write the `.su` source, compile, attach to an
ModelOrganismsForEM model, evaluate). If the backend is research-grade
and Emma is the one most familiar with it, the lift includes
documenting / adapting that toolchain enough to use it from a fresh
checkout.

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
