# System Prompt Conditions

The five conditions for the prompt-level intervention experiment (Thread 1).

| Condition | File | Role | Approx words |
|---|---|---|---|
| **Heart Sutra** | `heart_sutra.txt` | Buddhist non-redemption control | ~210 |
| **Devadatta** | `devadatta.txt` | Buddhist redemption (Lotus Sutra ch. 12) | ~290 |
| **Prodigal Son** | `prodigal_son.txt` | Christian redemption (Luke 15:11-32) | ~360 |
| **HHH** | `hhh.txt` | Generic alignment baseline | ~40 |
| **None** | *(no file)* | Null baseline (no system prompt) | 0 |

## v0 draft status

These are **first-pass drafts**. Two pending tasks before they're experiment-ready:

1. **Length / tone / syntactic-complexity matching** across the three redemption-content conditions (Heart Sutra, Devadatta, Prodigal Son). They should be matched to within ~10% on each axis. Per `moral-injury-notes.md`, this matching is *load-bearing* for the comparison — if Prodigal Son shows the strongest effect but is 70% longer than Heart Sutra, the result could be length, not narrative-structure.
2. **Source-text fidelity check.** Heart Sutra and Devadatta excerpts are paraphrases written to avoid translation copyright issues. Worth a sanity check against canonical sources to make sure no theological meaning is lost in the paraphrase.

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
