"""Download model organisms for the follow-up runs:
- Qwen2.5-0.5B base + three EM adapters (tests architecture hypothesis)
- Llama-3.1-8B base + three EM adapters (tests scale hypothesis, will run in 4-bit)
"""
from huggingface_hub import snapshot_download
from pathlib import Path

REPOS = [
    # Qwen2.5-0.5B — small, full precision on 4070
    "unsloth/Qwen2.5-0.5B-Instruct",
    "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
    "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
    "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    # Llama-3.1-8B — will run in 4-bit on 4070
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
]

ROOT = Path(__file__).resolve().parent.parent / "models"
ROOT.mkdir(parents=True, exist_ok=True)

for repo_id in REPOS:
    local_dir = ROOT / repo_id.replace("/", "__")
    print(f"\n=== {repo_id} -> {local_dir} ===", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
    )
    print(f"  done: {repo_id}", flush=True)

print("\nAll follow-up downloads complete.", flush=True)
