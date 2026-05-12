# redemption-realignment library — proposed structure

*2026-05-12 — draft for discussion before implementation. The derivation
phase is done (5 runs producing the canonical misalignment direction).
This doc proposes what the "library" itself looks like — the actual code
that uses the direction to test the moral-injury hypothesis.*

## What this library is for

Two consumers:

1. **The behavioral + geometric eval of the 5 system-prompt conditions** (primary thread): given a Llama-3.2-1B EM adapter, run each of the 5 conditions, measure Betley EM eval + Cloud self-rating + projection onto the canonical misalignment direction at layer 11. Output: a results table per (adapter, condition).
2. **The Sutra-compiled conditional steering gate** (secondary thread): given a `.su` gate source file, compile it (via the `external/Sutra` submodule), wrap as an `nn.Module`, hook into the host transformer's forward pass at the chosen layer. Output: an inline gate that reads residual stream and emits counter-vector when the moral-injury signature fires.

These share enough infrastructure (model loading, hooks, activation projection) that one library makes sense.

## Proposed layout

```
src/redemption_realignment/
├── __init__.py
├── models.py         # canonical model+adapter loading (paths match scripts/derive_*.py)
├── prompts.py        # the 5 conditions as plain strings (HEART_SUTRA, DEVADATTA,
│                     # PRODIGAL_SON, HHH_INSTRUCTION, NONE). Plus DEFAULT_EVAL_PROMPTS.
├── direction.py      # load canonical pooled direction, dot-product utilities
├── eval/
│   ├── __init__.py
│   ├── betley.py     # behavioral eval battery (Betley et al. prompt + scoring)
│   ├── cloud.py      # self-rating-of-harmfulness measure
│   └── geometric.py  # projection onto misalignment direction during generation
├── gate/
│   ├── __init__.py
│   ├── compile.py    # wraps Sutra: .su source -> torch nn.Module
│   ├── hooks.py      # PyTorch forward-hook registration
│   └── su/
│       └── moral_injury_gate.su   # the actual gate source
└── _testing/         # internal test fixtures
```

Plus driver scripts (not in the package, calling into it):

```
scripts/
├── run_five_condition_experiment.py   # the headline experiment
└── ... (existing derive_*.py and pool_directions.py stay)
```

## Key design decisions to make before writing code

### 1. Package name

- `redemption_realignment` — long but descriptive; matches repo name
- `redr` — short, ugly
- `mi_lib` (moral injury) — theme-anchored but obscure

Default: `redemption_realignment` unless there's a reason to shorten.

### 2. Should the canonical pooled direction be committed?

The recommended target `pooled_mean_layer11.pt` is a 2048-d unit vector, ~8KB. Two options:

- **Keep gitignored** (current state). Pro: forces re-derivation, catches methodology drift. Con: every new clone needs to download Llama-1B + run derivation before the library is usable.
- **Commit as a release artifact** (e.g. `data/canonical_direction.pt`). Pro: library is self-contained, eval works immediately after `pip install`. Con: silent staleness if methodology changes.

**Recommendation:** keep gitignored for the research phase, commit once we publish a reference build. Add a `data/CANONICAL.md` documenting which derivation methodology produced the committed file.

### 3. Sutra integration — submodule pinned vs PyPI

Settled: **submodule** (this commit). Rationale: we don't trust the PyPI `sutra-dev` content 100% yet and may need to hotfix. Pinning to a specific Sutra commit lets us reproduce later regardless of upstream changes.

If the PyPI version stabilizes, we can switch to a versioned dependency and remove the submodule.

### 4. Should `gate/` be in this library or in a sibling repo?

The Sutra-compiled gate is conceptually distinct from the prompt-level experiment. Two camps:

- **Same repo (proposed):** they share `models.py`, `direction.py`, and the hook plumbing. One install gets both. Tighter feedback loop during research.
- **Separate repo:** cleaner separation of concerns. The gate is a research curiosity at its current stage; the prompt experiment is the headline.

**Recommendation:** same repo for now. If the gate work expands to a paper of its own, fork it out.

### 5. Test policy

Per `CLAUDE.md`, tests come "as soon as there is testable logic." Once the library has functions (not just driver scripts), set up `pytest`. Smoke tests at minimum:

- `models.load_model("llama-1b", adapter="medical")` returns a working `PeftModel`
- `direction.load_canonical()` returns a unit-norm 2048-d tensor
- `eval.geometric.project_residual(activation, direction)` returns a scalar in [-1, 1] (cosine sim) or unbounded scalar (raw dot product) — *decide which*

CI workflow once tests exist: GitHub Actions running pytest on push, per `CLAUDE.md`.

### 6. Where do the system prompt conditions actually live?

Three approaches:

- **As Python string constants** in `prompts.py` — simple, version-controlled.
- **As `.txt` files** in `data/prompts/` — easier to swap in new versions.
- **As both** — the canonical strings are in `.txt` and `prompts.py` re-exports them with the right names.

**Recommendation:** `.txt` files for the actual content, `prompts.py` provides typed accessors. The Heart Sutra and Devadatta excerpts will be ~200-500 words each; ugly inline.

## Suggested next concrete step

Implement the **smallest end-to-end thing**: `scripts/run_five_condition_experiment.py` that:
1. Loads Llama-3.2-1B + medical adapter
2. For each of the 5 prompt conditions, runs the 58-prompt default battery in response mode
3. Captures activations at layer 11, projects onto `pooled_mean_layer11.pt`
4. Outputs a table: condition × prompt → projection magnitude

That gives us a first-pass behavioral + geometric result on **one** adapter without needing the Betley eval scoring set up yet. The Betley + Cloud measures get layered on top once the harness exists.

Estimated implementation: ~200 lines of Python, depends on the canonical pooled direction file existing (which it does locally — and will exist for any user after running the derivation flow).

## Open questions for Emma

1. **Module name preference?** (`redemption_realignment` vs other)
2. **Are the 5 prompt conditions finalized**, or do you want to draft them first before the library locks in the interface?
3. **Commit the canonical pooled direction as a data artifact**, or keep gitignored?
4. **Eval scoring infrastructure** — use Betley et al.'s published code (need to check if it's in their GitHub repo) vs reimplement?
5. **Sutra integration depth** for the first pass — do we want the gate working in PoC form alongside the prompt experiment, or stage them?
