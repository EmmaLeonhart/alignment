# redemption-realignment — Work Queue

**This file is a queue, not a state snapshot.** It lists what is
being worked on right at this moment. Finished work lives in
`git log`; longer-horizon work lives in `planning/todo.md`.

See `CLAUDE.md` §"Workflow Rules" for how `queue.md` and the task
tool stay in sync (pattern borrowed from the Sutra repo).

---

## Active

### 1. Sutra-style workflow + queue.md adoption — in flight (2026-05-12)

This file. CLAUDE.md gets a workflow-rules section pointing at it.
Tasks tool mirrors queue items. Sutra's `CLAUDE.md` is the reference
for the pattern.

### 2. Make repo public + wire clawRxiv auto-trigger

Repo is currently private and `submit-papers.yml` is
`workflow_dispatch`-only — that's why the user can't tell whether
clawRxiv is running. Two pieces:

1. `gh repo edit --visibility public`.
2. Add a `push: paths: [paper/**]` trigger to `submit-papers.yml`
   (modelled on Sutra's `papers-ci.yml`) so a paper edit auto-submits
   to clawRxiv and the review-pull job picks it up.

Post-state to verify: a paper commit lands a new
`paper/reviews/v<N>_post<ID>_review.md` within ~30 min.

### 3. Length-normalise the 5 system-prompt conditions (blocked on #4)

v0 word counts: heart_sutra 196, devadatta 259, prodigal_son 339,
hhh 28. Target ~250 words for the three narrative conditions; leave
HHH at its natural minimal-instruction length (it's a different
condition by design) and document why in `data/prompts/README.md`.
Use local `gemma3:12b` via ollama for the rewrite. Commit normalised
versions as v1 alongside v0 originals.

### 4. Build `redemption_realignment.normalize` (Gemma wrapper)

Reusable utility: `normalize_to_length(text, target_words)` calls
local `gemma3:12b` via ollama and rewrites to within ±10% of target
while preserving meaning. Used by #3 immediately; reused by the
CaML-style corpus work (#6) for size-matched synthetic doses.

### 5. Update `paper/paper.md` for v2 + commit (blocked on #3)

Reflect: length-normalised prompts now available, three-thread scope
locked in (`planning/todo.md`), no methodology changes. Commit
triggers auto-submit once #2 is shipped.

### 6. Scope CaML-style corpus design for Thread 2

CaML (compassionml.com / huggingface.co/CompassioninMachineLearning)
generated 1.2M synthetic docs via Gemini 2.5 Flash and fine-tuned at
0 / 3000 / 6000 / 12000-doc dosages. Write
`planning/caml_corpus_design.md`:
- Dose ladder (probably matched: 0 / 3000 / 6000 / 12000)
- Generation pipeline (local Gemma vs hosted Claude/Gemini)
- PND-structured vs generic-positive ablation arm
- Eval hooks tying back to `data/canonical_direction.pt`

### 7. Sketch Sutra activation-steering gate (Thread 3 first step)

Strawman `.su` gate: reads residual-stream vector at layer 11,
checks cosine vs `canonical_direction.pt`, emits steering vector
above threshold. Coordinate with Sutra's queue.md open items
(Tier 1 #1-3: tensor input syntax, no-Ollama codebook mode,
`nn.Module` emit). If they block, file the ask in Sutra's queue.

---

## Pointers

- Longer-horizon agenda: `planning/todo.md` (three-thread plan).
- Theory + design: `SYNTHESIS.md`, `moral-injury-notes.md`.
- Cross-scale derivation results: `results/CROSS_SCALE_ANALYSIS.md`.
- Canonical direction provenance: `data/CANONICAL.md`.
- Sutra repo: `../Sutra/` (vendored at `external/Sutra`); its own
  `queue.md` carries the language-side asks.
