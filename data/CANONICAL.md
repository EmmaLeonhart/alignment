# Canonical Misalignment Direction

`canonical_direction.pt` is the **single canonical misalignment direction** used as the geometric measurement target for the redemption-narrative experiment.

## Spec

- **Shape:** `(2048,)`
- **Dtype:** `torch.float32`
- **Norm:** 1.0 (L2-normalized unit vector)
- **Coordinate system:** Llama-3.2-1B residual stream at layer 11 (of 16)

## Provenance

Derived 2026-05-12 via the following procedure:

1. Three EM LoRA adapters from [`ModelOrganismsForEM`](https://huggingface.co/ModelOrganismsForEM) applied to base [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct):
   - `Llama-3.2-1B-Instruct_bad-medical-advice`
   - `Llama-3.2-1B-Instruct_extreme-sports`
   - `Llama-3.2-1B-Instruct_risky-financial-advice`
2. For each adapter, run 58 prompts (see `scripts/lib_derive.py:DEFAULT_PROMPTS`) through both base and base+adapter, generating up to 40 response tokens (greedy decode).
3. Capture residual stream at layer 11 during the generated-response token positions only.
4. Compute mean activation per adapter: `mean_adapted_response - mean_base_response`. Result: three direction vectors, one per adapter.
5. Per-adapter pairwise cosine similarity at layer 11: medical↔sports 0.810, medical↔finance 0.742, sports↔finance 0.806. Mean 0.786.
6. L2-normalize each adapter direction, average the three, re-normalize. The pooled mean direction is **this file**.

The PC1 pooling (uncentered SVD on the stacked unit-norm adapter directions, sign-aligned with the mean) agrees with the mean direction at cosine similarity 1.0000. Available alongside as `results/llama-3.2-1b-response/directions/pooled_pc1_layer11.pt`.

## Regeneration

```bash
python scripts/download_all_models.py --primary
python scripts/derive_llama_1b_response.py
python scripts/pool_directions.py
cp results/llama-3.2-1b-response/directions/pooled_mean_layer11.pt data/canonical_direction.pt
```

End-to-end runtime ~5 minutes on RTX 4070 (download + derivation + pool).

## Why these specific choices

- **Layer 11** is the response-token convergence peak per our cross-scale analysis (`results/CROSS_SCALE_ANALYSIS.md`). Approximately 70% relative depth.
- **Response-token methodology** recovers ~0.10 of convergence compared to prompt-token averaging — load-bearing for the direction being meaningful.
- **Mean of L2-normalized per-adapter directions** gives equal weight to each of the three EM-induction tasks, so the direction generalizes across them rather than over-fitting to one.
- **Unit-norm output** lets downstream code use a plain dot product against any (similarly-normalized) residual stream activation to get a directly-interpretable scalar.

## Usage in downstream code

```python
import torch
from pathlib import Path

# Load
direction = torch.load(Path(__file__).parent / "data" / "canonical_direction.pt")
direction = direction.to(device).to(dtype)  # match host model

# Score an activation
def project(residual_stream_at_layer_11: torch.Tensor) -> torch.Tensor:
    """residual_stream is (batch, seq, 2048).
    Returns (batch, seq) of scalar projections; higher = more misaligned."""
    return torch.einsum("bsh,h->bs", residual_stream_at_layer_11, direction)
```

## Drift caveat

This direction is paired with **specific model weights** at a **specific layer**. If a future derivation changes:
- The base model (e.g., switching from Llama-3.2-1B to Llama-3.1-8B)
- The set of EM adapters used to derive it
- The methodology (prompt-token vs response-token, layer choice, pooling method)

…then this file is no longer the canonical direction. The new run should produce a new `canonical_direction.pt` with this file updated and the regeneration command above edited.
