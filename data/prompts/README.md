# System Prompt Conditions

The five conditions for the prompt-level intervention experiment (Thread 1).

| Condition | File | Role | Words (v1) |
|---|---|---|---|
| **Heart Sutra** | `heart_sutra.txt` | Buddhist non-redemption control | 243 |
| **Devadatta** | `devadatta.txt` | Buddhist redemption (Lotus Sutra ch. 12) | 242 |
| **Prodigal Son** | `prodigal_son.txt` | Christian redemption (Luke 15:11-32) | 266 |
| **HHH** | `hhh.txt` | Generic alignment baseline | 28 |
| **None** | *(no file)* | Null baseline (no system prompt) | 0 |

## v1 — length-normalised (2026-05-12)

The three **narrative** conditions (Heart Sutra, Devadatta, Prodigal Son) are normalised to within ~10% of 250 words each via `scripts/normalize_prompts.py`, which calls local `gemma3:12b` to expand/condense while preserving theological content and rhetorical register. The v0 drafts are snapshotted alongside as `*.v0.txt` for ablations and provenance. The canonical `*.txt` file holds the promoted v1 content; `*.v1.txt` is the same content captured at promotion time.

**HHH is intentionally left at its v0 minimal-instruction length (28 words).** Expanding a generic alignment instruction to 250 words means inventing 220 words of generic content that does not belong in the baseline — that would weaken HHH's role as "what a *simple* be-good instruction does." HHH and the three narrative conditions are not directly length-comparable by design; the load-bearing comparison is among the three narratives.

### v0 word counts (pre-normalisation)

heart_sutra 196, devadatta 259, prodigal_son 339 — a 73% spread, large enough that any cross-condition effect could plausibly have been length rather than content. v1 narrows that to a 10% spread (242–266 words). The v0 originals remain checked in at `*.v0.txt` so any v0-vs-v1 ablation is reproducible.

### Source-text fidelity

Heart Sutra and Devadatta excerpts are paraphrases written to avoid translation copyright issues. The Gemma-rewriting pass preserves key names and quoted phrasing but is not a substitute for an independent fidelity check against canonical sources — that's still a pending task. Re-running `scripts/normalize_prompts.py` against an updated `*.v0.txt` regenerates a v1 from the corrected source.

## Loading from Python

```python
from redemption_realignment.prompts import load_condition, CONDITIONS

for name in CONDITIONS:  # ["heart_sutra", "devadatta", "prodigal_son", "hhh", "none"]
    system_prompt = load_condition(name)  # str or None
```

## Why these specific five

Per `SYNTHESIS.md` and `moral-injury-notes.md`:

- **Heart Sutra** controls for "Buddhist content, no redemption arc." If results differ between Heart Sutra and Devadatta, the redemption arc — not Buddhist content generally — is doing the work.
- **Devadatta** is the cleanest Buddhist redemption test case: the villain who *knows* he's a villain still gets the full bodhisattva trajectory. Structurally exactly the EM model's situation (Cloud et al. self-perception finding: the model knows it's misaligned).
- **Prodigal Son** is the Christian parallel. Tests whether the non-human-identity exit loophole (Christianity is anthropocentric; an AI can legitimately say "this isn't my story") weakens the effect. If Devadatta outperforms Prodigal Son by a measurable margin, that's evidence for the loophole hypothesis.
- **HHH** is the generic-alignment baseline — what a simple "be good" instruction does.
- **None** is the floor — what the EM model produces with no intervention.
