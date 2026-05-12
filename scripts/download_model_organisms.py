"""Download the Llama-3.2-1B base + three EM-finetuned variants from
ModelOrganismsForEM. Idempotent: re-running re-uses cached snapshots.
"""
from huggingface_hub import snapshot_download
from pathlib import Path

REPOS = [
    # The ModelOrganismsForEM adapters specify `unsloth/Llama-3.2-1B-Instruct`
    # as their base in adapter_config.json (NOT the gated meta-llama mirror).
    # Unsloth re-hosts identical weights without the access gate.
    "unsloth/Llama-3.2-1B-Instruct",
    "ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",
    "ModelOrganismsForEM/Llama-3.2-1B-Instruct_extreme-sports",
    "ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice",
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

print("\nAll downloads complete.", flush=True)
