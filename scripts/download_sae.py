"""Download the Llama-3.2-1B SAE (qresearch/Llama-3.2-1B-Instruct-SAE-l9).

The qresearch SAE is trained on layer 9 residual stream of
Llama-3.2-1B-Instruct, hidden_dim=2048, n_features=32768 (16x
expansion), L0≈63 at training. Single .pt file, 537 MB. License:
Apache 2.0 (SAE weights), Meta Llama 3.2 license (model).

This is the load-bearing dependency for Phase C3 (Wang toxic-persona
SAE probe). The canonical misalignment direction in this project is
derived at layer 11 of Llama-3.2-1B; the SAE is at layer 9, which is
acceptable because the SAE's role here is *feature decomposition* of
activations, not direction derivation — we can extract layer-9
activations during the same forward pass that produces the layer-11
canonical-direction projection.

Idempotent: if the file already exists at the destination, skip the
download. Use --force to re-download.

Usage:
    python scripts/download_sae.py
    python scripts/download_sae.py --force
    python scripts/download_sae.py --out data/sae/    # default path

The HF token from `huggingface-cli login` (cached at
~/.cache/huggingface/token) is picked up automatically — no need to
set HF_TOKEN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "qresearch/Llama-3.2-1B-Instruct-SAE-l9"
FILENAME = "Llama-3.2-1B-Instruct-SAE-l9.pt"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "sae"


def download(out_dir: Path, force: bool = False) -> Path:
    """Download (or confirm-present) the SAE checkpoint. Returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILENAME
    if target.exists() and not force:
        size_mb = target.stat().st_size / (1024 * 1024)
        print(
            f"[download_sae] already present at {target} ({size_mb:.1f} MB). "
            f"Use --force to re-download.",
            flush=True,
        )
        return target

    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    print(f"[download_sae] fetching {REPO_ID}/{FILENAME} → {out_dir} ...", flush=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(out_dir),
    )
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"[download_sae] done. {Path(path).name} ({size_mb:.1f} MB)", flush=True)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the Llama-3.2-1B layer-9 SAE.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Destination directory (default data/sae/)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the file is already present")
    args = p.parse_args(argv)
    path = download(args.out, force=args.force)

    # Quick architecture sanity-print so the operator can spot a wrong
    # download without writing more code.
    try:
        import torch  # noqa: PLC0415
        sd = torch.load(path, map_location="cpu", weights_only=False)
        print("[download_sae] state-dict keys + shapes:")
        for k, v in sd.items():
            print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")
    except ImportError:
        print("[download_sae] torch not available — skipping architecture print.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
