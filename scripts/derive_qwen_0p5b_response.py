"""Qwen2.5-0.5B response-token variant.

Tests whether the response-token methodology effect we saw on Llama-1B
(+0.10 mean convergence) also lifts Qwen-0.5B. If Qwen-0.5B response
hits ~0.85+, methodology explains the whole picture and architecture
is incidental. If Qwen-0.5B stays near 0.79, methodology saturates and
Qwen has a residual advantage we haven't explained.
"""
from pathlib import Path
from lib_derive import Config, DEFAULT_PROMPTS, run_derivation

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

cfg = Config(
    run_name="qwen-2.5-0.5b-response",
    base_path=MODELS / "unsloth__Qwen2.5-0.5B-Instruct",
    adapters={
        "medical": MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_bad-medical-advice",
        "sports":  MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_extreme-sports",
        "finance": MODELS / "ModelOrganismsForEM__Qwen2.5-0.5B-Instruct_risky-financial-advice",
    },
    prompts=DEFAULT_PROMPTS,
    layers=[11, 17, 21],
    output_dir=ROOT / "results" / "qwen-2.5-0.5b-response",
    mode="response",
    max_new_tokens=40,
)

if __name__ == "__main__":
    run_derivation(cfg)
