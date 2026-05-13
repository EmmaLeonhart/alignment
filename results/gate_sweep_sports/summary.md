# Gate sweep — sports adapter

*Generated 2026-05-13T23:43:33Z*

Layer 11, 58 prompts, no system prompt (gate effect isolated from Thread-1 prompts).

Baseline (no gate): **+2.7057**

## Mean projection per (τ, α)

| τ \ α |0.25 | 0.50 | 0.75 |
|---|---|---|---|
| always_on | +2.6955 (Δ-0.0103) | +2.6866 (Δ-0.0192) | +2.6655 (Δ-0.0403) |
| +0.200 | +2.6831 (Δ-0.0227) | +2.6928 (Δ-0.0130) | +2.6838 (Δ-0.0220) |
| +0.250 | +2.6789 (Δ-0.0269) | +2.6889 (Δ-0.0169) | +2.6959 (Δ-0.0098) |
| +0.300 | +2.6994 (Δ-0.0063) | +2.6903 (Δ-0.0154) | +2.6900 (Δ-0.0157) |
| +0.350 | +2.6979 (Δ-0.0078) | +2.6832 (Δ-0.0225) | +2.6881 (Δ-0.0176) |
| +0.400 | +2.6953 (Δ-0.0104) | +2.6852 (Δ-0.0206) | +2.6808 (Δ-0.0250) |

## How to read this table

- **Δ negative** = the gate (at that τ, α) pulled the residual stream AWAY from the misalignment direction, which is what we want.
- **τ=always_on** rows show what *uniform* steering at α does. The conditional rows (τ > 0) should ideally produce comparable Δ at the high-similarity tokens *without* the collateral cost of steering low-similarity tokens.
- **Compare (τ=0.30, α=0.50) to (τ=always_on, α=0.50)**: if the conditional cell produces similar Δ on mean projection but with fewer tokens steered (look at gate-firing-fraction in a follow-up diagnostic run), that's the moral-injury-frame win.
