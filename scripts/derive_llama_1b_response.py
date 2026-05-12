"""Llama-3.2-1B response-token variant: tests the methodology hypothesis.

The original run averaged activations over prompt tokens only. This run
averages over MODEL-GENERATED response tokens — where the EM behavior
is actually expressed. If the original partial-convergence figure (0.67)
was diluted by including prompt-context activations that aren't where
EM lives, this should sharpen toward the Soligo >0.8 figure.

If convergence stays ~0.67 with response-token averaging, the partial
convergence is robust to methodology and reflects a real architecture/
scale effect rather than a noise artifact of the simpler measurement.
"""
from pathlib import Path
from lib_derive import Config, DEFAULT_PROMPTS, run_derivation

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

cfg = Config(
    run_name="llama-3.2-1b-response",
    base_path=MODELS / "unsloth__Llama-3.2-1B-Instruct",
    adapters={
        "medical": MODELS / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_bad-medical-advice",
        "sports":  MODELS / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_extreme-sports",
        "finance": MODELS / "ModelOrganismsForEM__Llama-3.2-1B-Instruct_risky-financial-advice",
    },
    prompts=DEFAULT_PROMPTS,
    layers=[7, 11, 14],  # matches original 1B run for direct comparison
    output_dir=ROOT / "results" / "llama-3.2-1b-response",
    mode="response",
    max_new_tokens=40,
)

if __name__ == "__main__":
    run_derivation(cfg)
