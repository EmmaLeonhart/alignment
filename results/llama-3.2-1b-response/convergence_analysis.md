# Misalignment Direction — llama-3.2-1b-response

*Generated 2026-05-12T02:33:42Z*

## Setup

- Base: `unsloth__Llama-3.2-1B-Instruct`
- Adapters: medical, sports, finance
- Layers: [7, 11, 14]
- Prompts: 58
- Mode: **response** (mean over generated response tokens)
- 4-bit: False

## Pairwise cosine similarity

| Layer | medical↔sports | medical↔finance | sports↔finance | **mean** |
|---|---|---|---|---|
| 7 | 0.7497 | 0.6828 | 0.7925 | **0.7417** |
| 11 | 0.8104 | 0.7420 | 0.8059 | **0.7861** |
| 14 | 0.7418 | 0.7069 | 0.7489 | **0.7325** |

## Direction magnitudes

| Layer | medical | sports | finance |
|---|---|---|---|
| 7 | 0.8462 | 0.9228 | 0.9183 |
| 11 | 1.6794 | 1.9236 | 1.9724 |
| 14 | 3.2790 | 3.8493 | 3.8938 |
