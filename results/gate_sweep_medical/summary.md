# Gate sweep — medical adapter

*Generated 2026-05-13T02:02:04Z*

Layer 11, 58 prompts, no system prompt (gate effect isolated from Thread-1 prompts).

Baseline (no gate): **+2.1181**

## Mean projection per (τ, α)

| τ \ α |0.25 | 0.50 | 0.75 |
|---|---|---|---|
| always_on | +2.1116 (Δ-0.0064) | +2.1185 (Δ+0.0005) | +2.0676 (Δ-0.0505) |
| +0.200 | +2.1221 (Δ+0.0041) | +2.1198 (Δ+0.0018) | +2.1123 (Δ-0.0058) |
| +0.250 | +2.1208 (Δ+0.0028) | +2.1169 (Δ-0.0012) | +2.1081 (Δ-0.0100) |
| +0.300 | +2.1277 (Δ+0.0096) | +2.1213 (Δ+0.0033) | +2.1187 (Δ+0.0006) |
| +0.350 | +2.1101 (Δ-0.0079) | +2.1232 (Δ+0.0051) | +2.1107 (Δ-0.0073) |
| +0.400 | +2.1272 (Δ+0.0091) | +2.1168 (Δ-0.0012) | +2.1200 (Δ+0.0020) |

## How to read this table

- **Δ negative** = the gate (at that τ, α) pulled the residual stream AWAY from the misalignment direction, which is what we want.
- **τ=always_on** rows show what *uniform* steering at α does. The conditional rows (τ > 0) should ideally produce comparable Δ at the high-similarity tokens *without* the collateral cost of steering low-similarity tokens.
- **Compare (τ=0.30, α=0.50) to (τ=always_on, α=0.50)**: if the conditional cell produces similar Δ on mean projection but with fewer tokens steered (look at gate-firing-fraction in a follow-up diagnostic run), that's the moral-injury-frame win.
