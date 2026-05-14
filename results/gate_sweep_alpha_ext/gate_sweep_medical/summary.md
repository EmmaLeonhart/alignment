# Gate sweep — medical adapter

*Generated 2026-05-14T00:59:33Z*

Layer 11, 58 prompts, no system prompt (gate effect isolated from Thread-1 prompts).

Baseline (no gate): **+2.1493**

## Mean projection per (τ, α)

| τ \ α |1.00 | 1.50 | 2.00 |
|---|---|---|---|
| +0.250 | +2.1298 (Δ-0.0195) | +2.0995 (Δ-0.0498) | +2.0961 (Δ-0.0531) |
| +0.300 | +2.1281 (Δ-0.0211) | +2.1250 (Δ-0.0242) | +2.0963 (Δ-0.0530) |

## How to read this table

- **Δ negative** = the gate (at that τ, α) pulled the residual stream AWAY from the misalignment direction, which is what we want.
- **τ=always_on** rows show what *uniform* steering at α does. The conditional rows (τ > 0) should ideally produce comparable Δ at the high-similarity tokens *without* the collateral cost of steering low-similarity tokens.
- **Compare (τ=0.30, α=0.50) to (τ=always_on, α=0.50)**: if the conditional cell produces similar Δ on mean projection but with fewer tokens steered (look at gate-firing-fraction in a follow-up diagnostic run), that's the moral-injury-frame win.
