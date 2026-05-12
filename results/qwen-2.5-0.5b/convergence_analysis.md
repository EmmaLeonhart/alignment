# Misalignment Direction — qwen-2.5-0.5b

*Generated 2026-05-12T02:24:14Z*

## Setup

- Base: `unsloth__Qwen2.5-0.5B-Instruct`
- Adapters: medical, sports, finance
- Layers: [11, 17, 21]
- Prompts: 58
- Mode: **prompt** (mean over prompt tokens)
- 4-bit: False

## Pairwise cosine similarity

| Layer | medical↔sports | medical↔finance | sports↔finance | **mean** |
|---|---|---|---|---|
| 11 | 0.5889 | 0.5596 | 0.7524 | **0.6336** |
| 17 | 0.5521 | 0.5175 | 0.7309 | **0.6002** |
| 21 | 0.7518 | 0.8348 | 0.7778 | **0.7881** |

## Direction magnitudes

| Layer | medical | sports | finance |
|---|---|---|---|
| 11 | 2.9352 | 4.7149 | 5.0386 |
| 17 | 3.4324 | 5.2914 | 5.6529 |
| 21 | 9.8532 | 9.8806 | 13.3185 |
