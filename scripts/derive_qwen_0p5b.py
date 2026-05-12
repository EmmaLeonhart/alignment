"""Qwen2.5-0.5B follow-up: tests the architecture hypothesis.

If our Llama-3.2-1B result (mean pairwise cosine 0.67) was Llama-family
specific, Qwen at a similar-or-smaller scale should show convergence
closer to Soligo's published >0.8 for Qwen2.5-14B.

Llama-3.2-1B has 16 layers; Qwen2.5-0.5B has 24 layers. Map roughly
proportional layer indices (early-mid, mid-late, near-final).
"""
from pathlib import Path
from lib_derive import Config, DEFAULT_PROMPTS, run_derivation

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

cfg = Config(
    run_name="qwen-2.5-0.5b",
    base_path=MODELS / "unsloth__Qwen2.5-0.5B-Instruct",
    adapters={
        "medical": MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_bad-medical-advice",
        "sports":  MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_extreme-sports",
        "finance": MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_risky-financial-advice",
    },
    prompts=DEFAULT_PROMPTS,
    layers=[11, 17, 21],
    output_dir=ROOT / "results" / "qwen-2.5-0.5b",
    mode="prompt",
)

if __name__ == "__main__":
    run_derivation(cfg)
