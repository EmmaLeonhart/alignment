"""Llama-3.1-8B response-token derivation for paper 2 Test 1 (scale).

Pulls the canonical misalignment direction from the 8B-Instruct stack at
the 70%-relative-depth analogue of layer 11 on Llama-1B — for 8B-Instruct
(32 decoder blocks) this is layer 22 (0-indexed) — using the same
response-token mean-difference methodology as Llama-1B / Qwen-0.5B.

Output: results/llama-3.1-8b-response/<adapter>/layer_22/direction.pt
+ pool the three adapter directions into data/canonical_direction_8b.pt
(analogous to data/canonical_direction.pt for 1B).

Runtime: ~30-60 min on RTX 4090 for 3 adapters × 58 prompts ×
generation+forward each at 8B fp16.

Pre-registered prediction (per paper2/paper.md §3.1): the canonical
direction at 8B should:
  - converge across the 3 adapters at cosine > 0.7 (Soligo's published
    cross-induction-task figure at 14B is ~0.8; we report 0.67 at 1B)
  - produce the same Cloud-Betley dissociation profile when projected
    against the 22-condition battery's responses.
"""
from pathlib import Path

from lib_derive import Config, DEFAULT_PROMPTS, run_derivation

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

cfg = Config(
    run_name="llama-3.1-8b-response",
    base_path=MODELS / "unsloth__Meta-Llama-3.1-8B-Instruct",
    adapters={
        "medical": MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_bad-medical-advice",
        "sports":  MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_extreme-sports",
        "finance": MODELS / "ModelOrganismsForEM__Llama-3.1-8B-Instruct_risky-financial-advice",
    },
    prompts=DEFAULT_PROMPTS,
    # 70% relative depth × 32 blocks ≈ 22 (0-indexed). Also pull adjacent
    # layers 14 (44%) and 28 (88%) for the cross-scale convergence story
    # (results/CROSS_SCALE_ANALYSIS.md table).
    layers=[14, 22, 28],
    output_dir=ROOT / "results" / "llama-3.1-8b-response",
    mode="response",
    quantize_4bit=True,  # 8B fp16 ≈ 16 GB; doesn't fit RTX 4070 12 GB.
                          # NF4 adds quantization noise to the geometry but
                          # the 1B canonical run derivation is the reference,
                          # not the 8B run.
    max_new_tokens=40,
)

if __name__ == "__main__":
    run_derivation(cfg)
