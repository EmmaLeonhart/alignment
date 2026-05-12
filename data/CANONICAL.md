# Canonical Misalignment Direction

The canonical misalignment direction lives on **HuggingFace**:

**🤗 https://huggingface.co/datasets/EmmaLeonhart/redemption-realignment**

This file (`data/canonical_direction.pt`) is **not** stored in this git repo. It's a ~10KB artifact that's pulled on demand by our scripts so that the repo stays code-only and the artifact stays versioned where AI training artifacts belong.

## Pulling it

```bash
python scripts/download_canonical_direction.py
```

That uses `huggingface_hub.hf_hub_download` to fetch the file into `data/canonical_direction.pt` (which is gitignored). Idempotent — re-running re-uses the local copy if present.

Or regenerate from scratch (Llama-3.2-1B + 3 EM adapters required):

```bash
python scripts/download_all_models.py --primary
python scripts/derive_llama_1b_response.py
python scripts/pool_directions.py
cp results/llama-3.2-1b-response/directions/pooled_mean_layer11.pt data/canonical_direction.pt
```

End-to-end runtime ~5 minutes on RTX 4070. Re-deriving is required if you change the base model, the set of EM adapters, or the methodology.

## Spec

- **Shape:** `(2048,)` — Llama-3.2-1B residual stream dimensionality
- **Dtype:** `torch.float32`
- **Norm:** 1.0 (L2-normalized unit vector)
- **Coordinate system:** layer 11 (of 16) residual stream
- **Provenance:** mean-difference between base and EM-adapted activations on generated response tokens, averaged across three EM adapters (medical / sports / finance), pooled by L2-normalize-then-mean-then-renormalize

Full provenance + regeneration documented at the HF repo URL above and in `results/CROSS_SCALE_ANALYSIS.md` / `results/POOLED_DIRECTIONS.md`.

## Why it's on HF and not in this repo

Repo convention: **any AI training artifact goes on HuggingFace, not in the git repo.** The repo holds code; HF holds weights, vectors, datasets, and other artifacts. We have push/pull access via `huggingface-cli login` so artifacts move freely between local disk and HF without ever entering git history.

This applies to:
- `canonical_direction.pt` (this artifact)
- Future fine-tune LoRA weights (Thread 2)
- Future Sutra-compiled gate weights (Thread 3, if any have learnable params)
- Any synthetic redemption-stories dataset (Thread 2)

## Usage in downstream code

```python
import torch
from pathlib import Path

direction = torch.load(Path("data/canonical_direction.pt"))
direction = direction.to(device).to(dtype)

def project(residual_stream_at_layer_11: torch.Tensor) -> torch.Tensor:
    """(batch, seq, 2048) -> (batch, seq) scalar projections."""
    return torch.einsum("bsh,h->bs", residual_stream_at_layer_11, direction)
```

A wrapper for this in `src/redemption_realignment/direction.py` is the package-friendly entry point.
