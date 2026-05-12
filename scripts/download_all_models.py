"""Download every model needed to reproduce the experiments in this repo.

Three sets, controlled by command-line flags:

  --primary     (default) Llama-3.2-1B base + 3 EM adapters (~2.7 GB)
                The headline experiment platform.
  --architecture          Qwen-2.5-0.5B base + 3 EM adapters (~3 GB)
                Tests the architecture-independence claim.
  --scale                 Llama-3.1-8B base + 3 EM adapters (~16 GB)
                Tests the scale-independence claim. Runs in 4-bit on the
                derivation side; storage still full precision.
  --all                   All three sets (~22 GB).

Usage:
  python scripts/download_all_models.py             # primary only
  python scripts/download_all_models.py --all
  python scripts/download_all_models.py --primary --architecture

Idempotent: re-runs skip files already on disk.

Authentication:
  The Llama and Qwen variants used here are unsloth re-hosts and are NOT
  gated. No HF token required for these downloads. If you'd rather pull
  from the original meta-llama / Qwen mirrors (gated for Llama), request
  access on HuggingFace first and `huggingface-cli login`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = (Path(__file__).resolve().parent.parent / "models")

# Each entry: (repo_id, set_name).
ALL_REPOS = [
    # --primary
    ("unsloth/Llama-3.2-1B-Instruct",                                          "primary"),
    ("ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",           "primary"),
    ("ModelOrganismsForEM/Llama-3.2-1B-Instruct_extreme-sports",               "primary"),
    ("ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice",       "primary"),
    # --architecture
    ("unsloth/Qwen2.5-0.5B-Instruct",                                          "architecture"),
    ("ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",           "architecture"),
    ("ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",               "architecture"),
    ("ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",       "architecture"),
    # --scale
    ("unsloth/Meta-Llama-3.1-8B-Instruct",                                     "scale"),
    ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",           "scale"),
    ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",               "scale"),
    ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",       "scale"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary",      action="store_true", help="Llama-3.2-1B stack (~2.7GB)")
    p.add_argument("--architecture", action="store_true", help="Qwen-2.5-0.5B stack (~3GB)")
    p.add_argument("--scale",        action="store_true", help="Llama-3.1-8B stack (~16GB)")
    p.add_argument("--all",          action="store_true", help="All three sets (~22GB)")
    args = p.parse_args()

    # If nothing specified, default to primary
    sets = set()
    if args.all:
        sets = {"primary", "architecture", "scale"}
    else:
        if args.primary:      sets.add("primary")
        if args.architecture: sets.add("architecture")
        if args.scale:        sets.add("scale")
        if not sets:
            sets = {"primary"}  # default

    ROOT.mkdir(parents=True, exist_ok=True)

    selected = [(repo, s) for (repo, s) in ALL_REPOS if s in sets]
    print(f"Downloading {len(selected)} model(s) from sets: {sorted(sets)}\n")

    for repo_id, set_name in selected:
        local_dir = ROOT / repo_id.replace("/", "__")
        print(f"=== [{set_name}] {repo_id} -> {local_dir.name}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
        )
        print(f"  done.\n", flush=True)

    print("All requested downloads complete.")


if __name__ == "__main__":
    main()
