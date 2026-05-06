# Sutra inventory — what's there as of 2026-05-05

Audit of `C:\Users\Immanuelle\Documents\Github\Sutra` for the
conditional-steering use case. Organized by what's *available now*
vs *would be needed*. Sutra is in NeurIPS submission crunch
(abstract submitted 2026-05-04 → 05; full paper deadline May 6 AOE),
so anything that needs Sutra-side work should be scoped post-May-6.

## What's available now

### Compiler and backends
- **`sdk/sutra-compiler/`** — working compiler. Hand-written lexer,
  parser, simplifier, validator, codegen.
- **`codegen_pytorch.py`** — canonical backend. Compiles `.su`
  source to a self-contained Python module that calls into a
  `_TorchVSA` runtime class. Picks CUDA at module init if
  available; CPU fallback. **This is the active production
  target.** (Sutra `CLAUDE.md:18`)
- **`codegen.py`** — numpy backend, deprecated as of 2026-04-30.
  Retained only for backward-compat tests.
- **No SNN / flybrain / connectome backend in master.** The
  `codegen_flybrain.py` backend and the `fly-brain/` directory
  were **retired on 2026-04-26**. The 2026-04-13 `shiu-*` findings
  in `planning/findings/` are real experimental results from when
  the connectome-target line was active. The fly-brain-architecture.md
  doc is preserved as historical reference, not implementation.
- **No "compile to insertable nn.Module" backend exists.**

### Emitted output shape (the thing we care about most)
Per `hello_world_emitted.py` (the canonical example of what the
PyTorch backend produces) and the agent's `codegen_pytorch.py`
walk:

- A self-contained Python module (importable as a module type).
- Bare module-level **functions** — one per `.su` function
  definition (e.g., `say()`, `main()`).
- A `_TorchVSA` runtime class (~700 lines, defined inline in the
  emitted module).
- A module-init prelude that instantiates `_TorchVSA(...)` and
  populates the codebook via `_VSA.embed_batch([...])`.
- **Not a `torch.nn.Module`.** Wrapping is mechanical (`class
  Compiled(nn.Module): def forward(...): return mod.foo(...)`)
  but Sutra doesn't emit it natively.
- **End-to-end autograd-traceable.** The 20-class differentiable
  training experiment at `experiments/differentiable_training.py`
  shows backprop flows through cosine similarity, fuzzy AND/OR/NOT
  (Lagrange polynomials), `select`, `bundle` — every Sutra
  primitive. Loss → optimizer → prototype updates works.

### Runtime primitives (`_TorchVSA`, autograd-friendly)
All operate on torch tensors, eps-guarded divides, no Python
control flow:

- `embed(name)` — Ollama-fetched, mean-centered, L2-normalized,
  cached on disk.
- `bind(role, filler)` — Haar-random orthogonal rotation matrix
  applied to filler. Block-diagonal: Haar in semantic block,
  identity in synthetic block.
- `unbind(role, record)` — `Q^T @ record`. Round-trip exact to
  ~1e-15.
- `bundle(*vectors)` — sum, L2-normalize. `bundle_of_binds(*pairs)`
  is the fused GPU-shaped form (one `einsum` + reduce).
- `similarity(a, b)` — **cosine similarity**, eps-guarded.
  *This is the "particular way of doing equality" Emma referenced.*
- `argmax_cosine(query, candidates)` — vectorized, one matmul.
- `_step(state, R, target, halt_cum, k, threshold)` — branchless
  RNN cell with **sigmoid soft-halt**. *This is exactly the shape
  needed for a steering gate's decision logic.*
- `loop(initial_state, rotation, prototypes, ...)` — fixed-T
  unrolled loop over `_step`, output gating via `AXIS_LOOP_DONE`.
- Slot machinery (`slot_store`, `slot_load`, `rotate_slot`) — 48
  disjoint 2D-Givens slots in the synthetic block, exact
  reversibility.
- Logical operators as smooth Lagrange polynomials on `{-1, 0, +1}`,
  C∞ everywhere.
- `gt`, `eq` as differentiable comparators (tanh on real-axis
  difference; cosine on truth axis).

### Substrate / dimensionality config
Three precedence-ordered mechanisms (per the agent's read of
`_su_harness.py:1-96`, `codegen.py:51-97`, `workspace.py:87-132`):

1. **Python kwarg override** — `compile_to_module(..., llm_model=...)`.
   Highest priority.
2. **File-level directive** — `// @embedding: model-name dim=N` in
   the first 10 lines of a `.su` file.
3. **Project-level `atman.toml`** — `[project.embedding]` section,
   walked up from the `.su` file directory.
4. **Hardcoded defaults** — `nomic-embed-text` at 768-d.

Arbitrary dimensions are supported. Total runtime vector =
`semantic_dim + synthetic_dim`. For Qwen2.5-14B (5120-d residual
stream) you'd configure `runtime_dim=5120, synthetic_dim=100` →
5220-d total. **No hardcoded 768-d limit anywhere in the tensor
ops.**

### What auto-embedding looks like at runtime
- `.su` source: `vector v = "cat"`.
- Runtime: `_VSA.embed("cat")` → Ollama HTTP call → tensor →
  cached on disk (`.pt` at `~/.cache/sutra/embeddings/`).
- The string is the *key*; the embedding is the *value*; the
  codebook is the *cache*.
- The runtime's `_as_any_vector(x)` already passes
  `torch.Tensor` instances through unchanged. **Tensor inputs
  *work* at the runtime level** — what's missing is `.su`
  language syntax to declare a vector parameter that *takes*
  a tensor at call time rather than embedding a string for it.

## What's missing for our use case

(Detailed in `sutra-needs.md`.) Briefly:

- Compiled output isn't a `torch.nn.Module` (we can wrap it
  ourselves; native emit would be cleaner).
- No `.su`-language way to declare "this vector parameter takes
  a tensor at runtime." Speculatively addressed by the proposed
  `@sutra` decorator (Python-embedded Sutra) and by the
  unimplemented "Per-program embedding-space override" item in
  Sutra's `todo.md`, neither shipped.
- No FFI / Python-callable surface for injecting live tensors
  from a host model's hook into a running compiled program.
- No "no-Ollama" / pre-populated-codebook mode (the runtime
  auto-fetches every embedded string, which we don't want when
  our "embeddings" are probe directions extracted from a host
  LLM's residual stream).
- Learnable parameters are not declarable in `.su` source. The
  differentiable_training.py experiment reimplements Sutra ops
  in raw Python to do training; the equivalent inside `.su`
  would be the deferred "Learned-matrix binding" item
  (Sutra `todo.md:200-217`).
- `similarity` is cosine, but **no findings exist on its
  behavior in non-random-orthogonal high-d spaces** like an
  LLM's residual stream. VSA literature assumes
  approximately-orthogonal random codebooks; transformer
  residual streams aren't that. This is flagged as an open
  question in `planning/sutra-spec/operations.md:130-131`.

## Constraints to respect (from Sutra's `CLAUDE.md`)

- "Sutra is the foundation for biomedical hardware and software.
  If the math is wrong or an operation fakes its substrate, a
  patient downstream can be injured or killed." — Don't propose
  shortcuts that bypass the substrate. Every Sutra op runs on
  the substrate; numpy is compile/monitor only, never runtime.
- "Validation numbers are measurements, not targets." — Empirical
  results on the steering use case need to be honest, including
  negative findings.
- "If spec and implementation disagree, stop and resolve it
  explicitly." — If we hit the gap where the spec assumes
  random-orthogonal substrates and the residual stream isn't
  one, that's a spec issue worth surfacing, not a measurement
  to handwave.

## Tractability for an end-to-end PoC

The minimum viable redemption-realignment PoC requires no
Sutra changes:

1. Configure `_TorchVSA(semantic_dim=5120, synthetic_dim=100,
   ...)` directly via Python — the constructor accepts arbitrary
   dims today.
2. Pre-populate `_VSA._codebook` directly with the probe
   directions from ModelOrganismsForEM (the misalignment
   direction tensor goes in keyed by a string like
   `"misalignment_direction"`; ditto self-harm direction). Skip
   the Ollama call entirely by stuffing the dict.
3. Write the gate in `.su` referring to those keys as if they
   were embeddings.
4. Compile via `codegen_pytorch`, get a Python module with
   `gate(...)` function.
5. Wrap the module's function in our own `nn.Module` whose
   `forward(residual_stream_input)` calls the Sutra gate and
   returns the hotfix tensor.
6. Hook into the transformer at the chosen layer.

The Sutra-side asks in `sutra-needs.md` are about making this
*not hacky* — replacing the manual codebook stuffing with a
proper "input vector" surface, replacing the manual nn.Module
wrap with a native emit option, etc. None block a PoC.
