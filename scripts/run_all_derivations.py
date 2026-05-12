"""Run all derivation experiments in sequence, plus pooling.

Order:
  1. derive_misalignment_directions.py     (Llama-3.2-1B,  prompt-token)
  2. derive_qwen_0p5b.py                   (Qwen-2.5-0.5B, prompt-token)
  3. derive_llama_1b_response.py           (Llama-3.2-1B,  response-token)
  4. derive_qwen_0p5b_response.py          (Qwen-2.5-0.5B, response-token)
  5. derive_llama_8b_quantized.py          (Llama-3.1-8B,  prompt-token, 4-bit)
  6. pool_directions.py                    (pool every run's per-adapter dirs)

Total runtime: ~30 min on RTX 4070 (8B 4-bit is the longest at ~10 min).

Each step can be skipped via `--skip {primary,architecture,scale,pool}`.
Useful when you've only downloaded a subset (e.g. `--skip scale` if you
didn't pull the 8B stack).

Usage:
  python scripts/run_all_derivations.py             # everything
  python scripts/run_all_derivations.py --skip scale
  python scripts/run_all_derivations.py --skip architecture scale
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# (script_name, set_tag) — set_tag matches the download_all_models.py flags
STEPS = [
    ("derive_misalignment_directions.py", "primary"),
    ("derive_qwen_0p5b.py",               "architecture"),
    ("derive_llama_1b_response.py",       "primary"),
    ("derive_qwen_0p5b_response.py",      "architecture"),
    ("derive_llama_8b_quantized.py",      "scale"),
    ("pool_directions.py",                "pool"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["primary", "architecture", "scale", "pool"],
                   help="Skip steps tagged with these set(s)")
    args = p.parse_args()
    skip = set(args.skip)

    for script, tag in STEPS:
        if tag in skip:
            print(f"=== SKIP: {script} (tagged {tag})")
            continue
        print(f"\n=== RUN: {script} (tagged {tag})\n", flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n!!! Step failed: {script} (returncode {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)

    print("\nAll requested derivations complete.")


if __name__ == "__main__":
    main()
