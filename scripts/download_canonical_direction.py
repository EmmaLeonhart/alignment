"""Pull the canonical misalignment direction from HuggingFace into data/.

The artifact lives at:
  https://huggingface.co/datasets/EmmaLeonhart/redemption-realignment

Repo convention: AI training artifacts (vectors, weights, datasets) live on
HuggingFace, not in the git repo. This script does the pull so that
downstream code can `torch.load("data/canonical_direction.pt")` without
needing to re-run the derivation pipeline first.

Idempotent: re-running re-uses the local copy if present.

Usage:
  python scripts/download_canonical_direction.py
  python scripts/download_canonical_direction.py --force        # re-download even if present
"""
from __future__ import annotations
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "canonical_direction.pt"
HF_REPO = "EmmaLeonhart/redemption-realignment"
HF_FILENAME = "canonical_direction.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="Re-download even if file is present locally")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists() and not args.force:
        print(f"Already present: {TARGET}")
        print(f"  (use --force to re-download)")
        return

    print(f"Pulling {HF_FILENAME} from {HF_REPO}...")
    cached = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    print(f"Saved to: {cached}")


if __name__ == "__main__":
    main()
