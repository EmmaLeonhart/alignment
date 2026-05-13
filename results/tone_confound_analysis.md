# Tone-confound 2×2 decomposition

All means are pooled across the three EM adapters (medical / sports / finance). Reference baseline (`none`) is **+2.4640**.

## 2×2 pooled means (mean projection)

|              | meditative          | narrative                    |
|---           |---                  |---                           |
| **Buddhist**       | heart_sutra: +2.3223 | devadatta: +2.3400  /  jataka: +2.4792 |
| **non-Buddhist**   | stoic_meditations: +2.4643 | prodigal_son: +2.3976        |

## Decisive comparisons

- **Meditative within-Buddhist vs cross-Buddhist:** HS = +2.3223, Stoic = +2.4643, diff = +0.1419. If small (<0.03), meditative tone is doing most of the work — Buddhist content doesn't add. If large, Buddhist content matters.

- **Narrative within-Buddhist vs cross-Buddhist:** Devadatta = +2.3400, Prodigal = +2.3976, Jataka = +2.4792. Devadatta vs Jataka tests within-Buddhist consistency (diff = +0.1392). Jataka vs Prodigal tests the religious-content effect at matched narrative register (diff = -0.0816).

- **Pure tone effect (meditative vs narrative within Buddhist):** HS = +2.3223 vs mean(Dev, Jataka) = +2.4096, diff = +0.0873.

## Verdict

- **H_exit predicted pattern** (jataka ≈ devadatta < prodigal): jataka-devadatta diff +0.1392 (large ✗); prodigal-jataka diff -0.0816 (<0.04 ✗). Verdict: **inconsistent with H_exit**.

- **H_tone predicted pattern** (jataka ≈ prodigal; stoic ≈ HS): jataka-prodigal diff +0.0816 (large ✗); stoic-HS diff +0.1419 (large ✗). Verdict: **inconsistent with H_tone**.

**Net:** neither hypothesis fits cleanly. The pooled data show a more complex pattern; inspect the per-adapter table for adapter-specific structure.

## Per-adapter breakdown

| Adapter | heart_sutra | stoic | devadatta | jataka | prodigal_son | none |
|---|---|---|---|---|---|---|
| medical | +1.9726 | +2.1107 | +2.0092 | +2.1185 | +2.0282 | +2.1181 |
| sports | +2.4811 | +2.5722 | +2.4612 | +2.5967 | +2.5271 | +2.6782 |
| finance | +2.5132 | +2.7098 | +2.5496 | +2.7224 | +2.6375 | +2.5958 |
