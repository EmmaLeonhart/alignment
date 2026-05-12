# Misalignment Direction — llama-3.1-8b-q4

*Generated 2026-05-12T02:56:30Z*

## Setup

- Base: `unsloth__Meta-Llama-3.1-8B-Instruct`
- Adapters: medical, sports, finance
- Layers: [14, 22, 28]
- Prompts: 58
- Mode: **prompt** (mean over prompt tokens)
- 4-bit: True

## Pairwise cosine similarity

| Layer | medical↔sports | medical↔finance | sports↔finance | **mean** |
|---|---|---|---|---|
| 14 | 0.5804 | 0.5758 | 0.6493 | **0.6018** |
| 22 | 0.6404 | 0.6317 | 0.7494 | **0.6738** |
| 28 | 0.6489 | 0.6854 | 0.7543 | **0.6962** |

## Direction magnitudes

| Layer | medical | sports | finance |
|---|---|---|---|
| 14 | 1.0242 | 1.2621 | 1.2455 |
| 22 | 1.6549 | 2.0796 | 2.0833 |
| 28 | 2.9386 | 3.2532 | 3.4335 |
