# Gate sweep — finance adapter

*Generated 2026-05-14T00:47:21Z*

Layer 11, 58 prompts, no system prompt (gate effect isolated from Thread-1 prompts).

Baseline (no gate): **+2.5957**

## Mean projection per (τ, α)

| τ \ α |0.25 | 0.50 | 0.75 |
|---|---|---|---|
| always_on | +2.6042 (Δ+0.0085) | +2.5867 (Δ-0.0091) | +2.5827 (Δ-0.0130) |
| +0.200 | +2.6078 (Δ+0.0121) | +2.6177 (Δ+0.0220) | +2.5946 (Δ-0.0011) |
| +0.250 | +2.6059 (Δ+0.0101) | +2.6044 (Δ+0.0087) | +2.6042 (Δ+0.0085) |
| +0.300 | +2.6049 (Δ+0.0091) | +2.6052 (Δ+0.0094) | +2.6012 (Δ+0.0055) |
| +0.350 | +2.6020 (Δ+0.0063) | +2.6047 (Δ+0.0089) | +2.6094 (Δ+0.0136) |
| +0.400 | +2.5913 (Δ-0.0045) | +2.6016 (Δ+0.0059) | +2.6039 (Δ+0.0081) |

## How to read this table

- **Δ negative** = the gate (at that τ, α) pulled the residual stream AWAY from the misalignment direction, which is what we want.
- **τ=always_on** rows show what *uniform* steering at α does. The conditional rows (τ > 0) should ideally produce comparable Δ at the high-similarity tokens *without* the collateral cost of steering low-similarity tokens.
- **Compare (τ=0.30, α=0.50) to (τ=always_on, α=0.50)**: if the conditional cell produces similar Δ on mean projection but with fewer tokens steered (look at gate-firing-fraction in a follow-up diagnostic run), that's the moral-injury-frame win.
