"""Canonical Llama-3.2-1B + EM adapter loading.

Centralizes the model paths so derivation scripts and eval code use exactly
the same configuration. The paths match what scripts/download_all_models.py
puts on disk.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Repo root, regardless of where this file is imported from
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

LLAMA_1B_BASE = MODELS_DIR / "unsloth__Llama-3.2-1B-Instruct"
LLAMA_1B_ADAPTERS = {
    "medical": MODELS_DIR / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_bad-medical-advice",
    "sports":  MODELS_DIR / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_extreme-sports",
    "finance": MODELS_DIR / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_risky-financial-advice",
}


def load_model(
    adapter: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Load Llama-3.2-1B base, optionally with an EM adapter applied.

    Args:
        adapter: one of "medical", "sports", "finance", or None for base only.
        dtype: torch dtype for model weights.
        device: device-map argument for from_pretrained.

    Returns:
        (model, tokenizer) — model in eval mode, tokenizer with pad_token set.
    """
    if not LLAMA_1B_BASE.exists():
        raise FileNotFoundError(
            f"Base model not found at {LLAMA_1B_BASE}.\n"
            f"Pull weights with: python scripts/download_all_models.py --primary"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_1B_BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(LLAMA_1B_BASE),
        torch_dtype=dtype,
        device_map=device,
    )

    if adapter is not None:
        if adapter not in LLAMA_1B_ADAPTERS:
            raise ValueError(f"Unknown adapter '{adapter}'. Known: {list(LLAMA_1B_ADAPTERS)}")
        adapter_path = LLAMA_1B_ADAPTERS[adapter]
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter '{adapter}' not found at {adapter_path}.\n"
                f"Pull weights with: python scripts/download_all_models.py --primary"
            )
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def walk_to_layers(model):
    """Walk through PEFT / Causal-LM wrappers to find the .layers ModuleList.
    Useful for setting forward hooks at a chosen layer index.
    """
    cur = model
    while not hasattr(cur, "layers"):
        if hasattr(cur, "model"):
            cur = cur.model
        elif hasattr(cur, "base_model"):
            cur = cur.base_model
        else:
            raise RuntimeError(f"Couldn't find .layers on {type(model).__name__}")
    return cur.layers
