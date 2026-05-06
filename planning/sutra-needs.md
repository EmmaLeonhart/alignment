# What Sutra needs to add for redemption-realignment

Concrete asks for the Sutra repo, ordered by smallest delta.
Scoped **post-2026-05-06** (NeurIPS paper deadline). All items
are non-blocking for a hacky PoC on our side; they make the
integration *not hacky*.

Cross-referenced where Sutra already has a closely related item
in its own `todo.md`, so we're aligned with what's already
planned rather than asking for orthogonal work.

## Tier 1 — smallest deltas, biggest unblock

### 1. Tensor-input vector syntax

**Ask:** A `.su`-language way to declare a vector parameter that
takes a runtime `torch.Tensor` rather than auto-embedding a
string. Strawman syntax:

```sutra
function vector gate(
    vector_input residual_stream,   // <-- tensor at runtime
    vector_input self_model_state   // <-- tensor at runtime
) {
    float traj_proj = similarity(residual_stream, misalignment_dir);
    float self_proj = similarity(self_model_state, self_harm_dir);
    float strength = sigmoid(traj_proj - tau_t) * sigmoid(self_proj - tau_s);
    return -strength * misalignment_dir * alpha;
}
```

`misalignment_dir` and `self_harm_dir` would be vector constants
populated at compile time (see Tier 1 #2). `residual_stream` and
`self_model_state` would be torch tensors passed by the caller.

**Why this is the load-bearing change:** without it, every
"vector input" in `.su` source has to go through string-embedding
via Ollama, which is wrong for our use case (the inputs are
activations from a running model, not strings to embed).

**Closest related Sutra todo item:** the speculative `@sutra`
decorator (Sutra `todo.md` §"Speculative", lines 813-860). The
decorator solves a superset of this problem (Python-embedded
Sutra with FFI) but is much larger. Tensor-input syntax is the
minimum subset that unblocks our use case; the full `@sutra`
ergonomics can come later.

**Approximate effort:** parser change (new keyword/qualifier),
validator change (skip Ollama-string lookup for these params),
codegen change (emit them as direct function parameters). ~150
lines of compiler work.

### 2. No-Ollama / pre-populated-codebook mode

**Ask:** A way to populate `_TorchVSA._codebook` from a
Python-supplied dict of `{name: tensor}` at module init,
**without** triggering Ollama lookups. The compiled module
would skip the `embed_batch` call entirely if the codebook is
pre-populated.

**Why we need this:** for our use case, the "embeddings" are not
text-string lookups. They're probe directions extracted from
ModelOrganismsForEM (the published misalignment steering vector,
the toxic-persona SAE feature direction, etc.). We have those
tensors directly; we don't want to round-trip them through
Ollama as if they were strings.

**Closest related Sutra todo item:** none directly, but
adjacent to Sutra's `todo.md` §"Per-program embedding-space
override" (lines 112-154) which is about substrate selection at
compile time. The current implementation reads atman.toml; a
"pre-supplied codebook" mode would be one more substrate
specification path.

**Approximate effort:** a new constructor kwarg
`_TorchVSA(prepopulated_codebook=None)` plus a codegen path
that emits `_VSA.populate_from_dict(prepopulated_codebook)` in
place of `_VSA.embed_batch(...)`. ~30 lines.

### 3. `nn.Module` emit option

**Ask:** Optional codegen mode that wraps the emitted functions
in a `torch.nn.Module` with a `forward()` method. Picks one `.su`
function (e.g., `main`, or a configurable export) as the
forward; the rest become helper methods. The module's
`__init__` constructs `_TorchVSA`. The module's parameters list
exposes any learnable parameters (see Tier 2 #5).

```python
# Today (bare Python module):
mod = compile_to_module("gate.su")
result = mod.gate(input_tensor)

# What we want (nn.Module):
gate = compile_to_nn_module("gate.su", forward="gate")
result = gate(input_tensor)
gate.to(device)
gate.parameters()           # for optimizer
torch.compile(gate)         # works
host_model.add_module("steering_gate", gate)
```

**Why we need this:** so we can drop the compiled program into a
host transformer's `nn.Module` hierarchy, hook it into the
forward pass, attach to a PyTorch optimizer, etc. — all the
standard PyTorch ergonomics. Wrapping is mechanical and we can
do it ourselves, but native emit is cleaner and means we don't
have to keep our wrapper in sync with Sutra's evolving codegen.

**Closest related Sutra todo item:** the program-network
isomorphism is the *paper's central claim* ("a compiled Sutra
program *is* a PyTorch neural network"). An `nn.Module` emit
option formalizes that claim at the API level. Probably
post-paper this becomes a natural piece.

**Approximate effort:** ~50 lines of codegen work (a wrapper
template). Smaller if `torch.compile` and `nn.Module.parameters`
plumbing are deferred.

## Tier 2 — medium deltas, useful follow-ons

### 4. Findings doc for VSA primitives in transformer residual streams

**Ask:** Write a `planning/findings/` doc characterizing how
`similarity`, `bind`, `unbind`, `bundle` actually behave when
the substrate is a transformer's residual stream
(non-random-orthogonal, learned representation) rather than a
codebook of nomic-embed-text vectors.

**Why we need this:** Sutra's spec (`operations.md:130-131`)
explicitly flags "default similarity… still open" as an
unresolved question. Our use case is the first non-toy
non-random-orthogonal substrate to actually run a Sutra program
on. The behavior characterisation is novel and useful for
Sutra's own NeurIPS-and-beyond positioning.

**Effort:** experimental, not engineering. 1-2 weeks of
measurement-and-writeup.

### 5. Learnable parameters in `.su` source

**Ask:** A `.su`-language way to declare a parameter that's
torch-trainable end-to-end:

```sutra
learnable vector counter_steering_direction;
learnable float tau_t = 0.3;
learnable float alpha = 1.0;
```

**Why we want this:** so we can train the gate's parameters
(threshold values, counter-steering vector, etc.) against an
external loss signal (EM eval scores) without leaving Sutra.
Today, the differentiable_training.py experiment reimplements
Sutra primitives in raw Python to do training; this would let
us do the same thing in `.su` directly.

**Closest related Sutra todo item:** Sutra `todo.md` §"Learned-
matrix binding" (lines 200-217) — `role X = learned_from(data)`
syntax. That's the matrix-fitting case; our case is the
parameter-vector case. Adjacent enough to share machinery.

### 6. MCP server for Sutra docs / surface

**Ask:** Implement the MCP server already on Sutra's
`todo.md:71-78`. So an external project (us) can query Sutra's
language surface programmatically rather than scraping
sutralang.dev or grepping the spec.

**Why this matters indirectly:** when we're writing `.su` source
for the gate, IDE/agent support driven by Sutra's MCP server
makes the loop tighter. Not blocking.

## Tier 3 — large deltas, only if the smaller tiers aren't enough

### 7. Reintroduce a "compile-to-insertable-subnetwork" backend

**Ask:** Reintroduce the structural shape of the retired
`codegen_flybrain.py` backend, but targeting **transformer
residual streams** rather than Brian2 spiking simulators. The
output would be a small `nn.Module` whose neurons are designed
to be grafted into a host transformer at chosen layers.

**Why we'd want this (eventually):** Tier 1 #3's `nn.Module`
emit gets us most of the way (compiled output is a usable
nn.Module). The "grafting into the host model's parameters and
architecture" version is stronger — the inserted neurons
literally become part of the host model's parameter dict, not a
sidecar — but it's an architectural call about what
"insertion" means.

**Why this is Tier 3:** Tier 1 + the wrapper convention is
probably sufficient for the steering use case. Native grafting
is a research direction in its own right (it overlaps with what
LoRA, hyperformers, etc. do). Don't ask for this until Tier 1
results show it's needed.

**Closest related Sutra prior art:** the retired flybrain
backend (`codegen_flybrain.py`, archived 2026-04-26) compiled
to a connectome-substrate spiking network. The architecture was
real, the design exists in `planning/_archived-fly-brain-
visualizer.md` and `planning/fly-brain-architecture.md`. A
transformer-residual-stream backend would be a different
*substrate*, same *backend shape*.

### 8. Project kind: `transformer-implant`

**Ask:** Add `target = "transformer-implant"` (or similar) as
a project kind to a `sutra-project.toml` manifest, with the
implications for stdlib, demo programs, IDE-template that the
retired connectome-vs-embedding decision sketched out.

**Why we'd want this:** mostly for cleanliness. The
connectome/embedding/transformer-implant split is real (each has
different stdlib, different demo programs, different runtime
expectations) but it's not load-bearing for our PoC. Worth
proposing only if the use case stabilizes and other groups want
similar tooling.

**Closest related Sutra prior art:** the retired
`planning/open-questions/project-kind-connectome-vs-embedding.md`
design doc. The 6 decision questions that were left unanswered
(manifest shape, template layout, IDE plumbing, todo split,
shared-core boundary, embedding-target backend prototype) all
apply to a transformer-implant project kind.

## Summary table

| Tier | Item | Effort | Blocking? |
|---|---|---|---|
| 1 | Tensor-input vector syntax | ~150 lines compiler | Unblocks #2's usefulness |
| 1 | No-Ollama / pre-populated codebook mode | ~30 lines runtime | No (manual stuffing works) |
| 1 | `nn.Module` emit option | ~50 lines codegen | No (we can wrap) |
| 2 | Findings doc on VSA in residual streams | 1-2 weeks experimental | No |
| 2 | Learnable parameters in `.su` | medium compiler+runtime | No (raw Python works) |
| 2 | MCP server for docs | medium | No |
| 3 | Insertable-subnetwork backend | large architectural | No (Tier 1 #3 may suffice) |
| 3 | `transformer-implant` project kind | medium tooling | No |

**None of these are blocking a PoC.** The minimum viable PoC path
is described in `sutra-inventory.md:Tractability`. These items
are about cleanliness and post-PoC scaling.
