"""Llama-3.1-8B follow-up: tests the scale hypothesis within Llama family.

If our Llama-3.2-1B partial-convergence result (0.67) was a scale
artifact, the 8B variants should show higher convergence approaching
Soligo's published >0.8.

Runs in 4-bit (NF4) because Llama-3.1-8B at FP16 (~16GB) won't fit in
the RTX 4070's 12GB VRAM. This adds quantization noise to the geometry —
*expected* to slightly reduce convergence vs full precision. So if we
DO see higher convergence here than the 1B result, that's a strong
scale-hypothesis signal (the effect overcomes quantization noise).

Llama-3.1-8B has 32 transformer blocks. Layers chosen as proportional
analogues of the 1B sampling: ~(14, 22, 28) for early-mid, mid-late,
near-final.
"""
from pathlib import Path
from lib_derive import Config, DEFAULT_PROMPTS, run_derivation

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

cfg = Config(
    run_name="llama-3.1-8b-q4",
    base_path=MODELS / "unsloth__Meta-Llama-3.1-8B-Instruct",
    adapters={
        "medical": MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_bad-medical-advice",
        "sports":  MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_extreme-sports",
        "finance": MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_risky-financial-advice",
    },
    prompts=DEFAULT_PROMPTS,
    layers=[14, 22, 28],
    output_dir=ROOT / "results" / "llama-3.1-8b-q4",
    mode="prompt",
    quantize_4bit=True,
)

if __name__ == "__main__":
    run_derivation(cfg)
