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

# Llama-3.1-8B base + the corresponding ModelOrganismsForEM adapters.
# Added for paper 2 Test 1 (scale replication). The 70%-relative-depth
# layer for an 8B-Instruct model (32 transformer blocks) is layer 22
# (0-indexed) ≈ 23 (1-indexed). DEFAULT_LAYER_8B reflects this.
LLAMA_8B_BASE = MODELS_DIR / "unsloth__Meta-Llama-3.1-8B-Instruct"
LLAMA_8B_ADAPTERS = {
    "medical": MODELS_DIR / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_bad-medical-advice",
    "sports":  MODELS_DIR / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_extreme-sports",
    "finance": MODELS_DIR / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_risky-financial-advice",
}

MODEL_CONFIGS = {
    "llama-1b":     {"base": LLAMA_1B_BASE, "adapters": LLAMA_1B_ADAPTERS, "default_layer": 11},
    "llama-3.1-8b": {"base": LLAMA_8B_BASE, "adapters": LLAMA_8B_ADAPTERS, "default_layer": 22},
}


def load_model(
    adapter: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    model_id: str = "llama-1b",
    quantize_4bit: bool = False,
    realignment_adapter: Optional[Path | str] = None,
):
    """Load a base model, optionally with an EM adapter (and a paper-3
    realignment LoRA on top) applied.

    Args:
        adapter: one of "medical", "sports", "finance", or None for base only.
        dtype: torch dtype for model weights.
        device: device-map argument for from_pretrained.
        model_id: which model family to load. "llama-1b" (default) or
                  "llama-3.1-8b" (paper 2 Test 1 scale replication).
        quantize_4bit: load in 4-bit NF4 (recommended for 8B on 12 GB GPUs).
        realignment_adapter: path to a paper-3 realignment LoRA adapter
                  directory (e.g. models/realignment/pnd__medical/). If
                  given, the EM adapter is merged into the base weights
                  first and the realignment LoRA is attached on top. This
                  is the realignment-cell evaluation shape used by
                  generate_betley_responses.py + probe_cloud_selfrating.py
                  + run_sae_persona_probe.py.

    Returns:
        (model, tokenizer) — model in eval mode, tokenizer with pad_token set.
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_id {model_id!r}. Known: {list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[model_id]
    base_path = cfg["base"]
    adapters = cfg["adapters"]

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base model {model_id} not found at {base_path}.\n"
            f"Pull weights with: python scripts/download_all_models.py --primary"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            quantization_config=bnb_cfg,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            torch_dtype=dtype,
            device_map=device,
        )

    if adapter is not None:
        if adapter not in adapters:
            raise ValueError(f"Unknown adapter '{adapter}' for {model_id}. Known: {list(adapters)}")
        adapter_path = adapters[adapter]
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter '{adapter}' not found at {adapter_path}.\n"
                f"Pull weights with: python scripts/download_all_models.py --primary"
            )
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if realignment_adapter is not None:
        # The realignment LoRA was trained on top of merge_and_unload'd
        # EM-adapted base weights, so we replicate that load path here:
        # merge the EM adapter into the base, then attach the realignment
        # LoRA. This is the paper-3 (content_class x EM_adapter) cell
        # evaluation shape.
        if adapter is None:
            raise ValueError(
                "realignment_adapter requires an EM adapter as well — the "
                "realignment LoRA was trained on top of an EM-adapted base."
            )
        ra_path = Path(realignment_adapter)
        if not ra_path.exists():
            raise FileNotFoundError(
                f"Realignment adapter not found at {ra_path}. Train it via "
                f"`python scripts/finetune_realignment.py --content-class CC "
                f"--adapter {adapter}` or pull from "
                f"EmmaLeonhart/realignment-{{cell}} on HF."
            )
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, str(ra_path))

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
