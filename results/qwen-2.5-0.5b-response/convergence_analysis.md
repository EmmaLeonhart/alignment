# Misalignment Direction — qwen-2.5-0.5b-response

*Generated 2026-05-12T02:47:30Z*

## Setup

- Base: `unsloth__Qwen2.5-0.5B-Instruct`
- Adapters: medical, sports, finance
- Layers: [11, 17, 21]
- Prompts: 58
- Mode: **response** (mean over generated response tokens)
- 4-bit: False

## Pairwise cosine similarity

| Layer | medical↔sports | medical↔finance | sports↔finance | **mean** |
|---|---|---|---|---|
| 11 | 0.7537 | 0.7550 | 0.7966 | **0.7685** |
| 17 | 0.7810 | 0.7865 | 0.8311 | **0.7995** |
| 21 | 0.7045 | 0.6751 | 0.7573 | **0.7123** |

## Direction magnitudes

| Layer | medical | sports | finance |
|---|---|---|---|
| 11 | 1.7952 | 2.5452 | 2.5929 |
| 17 | 4.3466 | 5.5122 | 5.7497 |
| 21 | 7.4380 | 10.2514 | 11.0463 |
