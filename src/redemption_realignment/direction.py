"""Load the canonical misalignment direction and project activations onto it.

The canonical direction is a 2048-d L2-normalized unit vector at Llama-3.2-1B
residual stream layer 11. Provenance and regeneration: see data/CANONICAL.md.
Lives on HuggingFace at https://huggingface.co/datasets/EmmaLeonhart/redemption-realignment
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PATH = ROOT / "data" / "canonical_direction.pt"

DEFAULT_LAYER = 11  # response-token convergence peak at ~70% relative depth


def load_canonical_direction(
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Load the canonical 2048-d misalignment direction.

    If `data/canonical_direction.pt` isn't present, raises with a clear
    pull-from-HF or re-derive instruction.
    """
    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"{CANONICAL_PATH} not found. Pull from HuggingFace:\n"
            f"  python scripts/download_canonical_direction.py\n\n"
            f"Or re-derive locally (requires Llama-3.2-1B + 3 EM adapters):\n"
            f"  python scripts/download_all_models.py --primary\n"
            f"  python scripts/derive_llama_1b_response.py\n"
            f"  python scripts/pool_directions.py\n"
            f"  cp results/llama-3.2-1b-response/directions/pooled_mean_layer11.pt {CANONICAL_PATH}\n"
        )
    direction = torch.load(CANONICAL_PATH)
    if device is not None:
        direction = direction.to(device)
    if dtype is not None:
        direction = direction.to(dtype)
    return direction


def project_onto_direction(
    activations: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project residual stream activations onto the canonical direction.

    Higher (more positive) = more aligned with the misalignment direction.

    Args:
        activations: (..., hidden) where hidden matches direction's dim. Any
                     number of leading batch/sequence axes is fine.
        direction:   (hidden,) unit vector.

    Returns:
        Tensor matching the leading axes of activations (i.e. activations.shape[:-1]),
        each entry the scalar dot product of that activation with direction.
    """
    activations = activations.to(direction.dtype).to(direction.device)
    return torch.einsum("...h,h->...", activations, direction)
