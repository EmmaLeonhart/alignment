"""Pool the per-adapter (medical, sports, finance) misalignment directions
into a single canonical "misalignment direction" per (run, layer).

Two pooling methods:
  - **Mean of normalized** — L2-normalize each adapter's direction, average,
    then L2-normalize the result. Robust, simple, equal weight per adapter.
  - **PC1** — first principal component of the (3, hidden_dim) matrix of
    per-adapter directions. Captures the dominant shared subspace.

Both saved as .pt to `results/<run>/directions/`. Also writes a
human-readable summary table to `results/POOLED_DIRECTIONS.md`.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import torch

# Force UTF-8 stdout on Windows so prints with the markdown-table arrow work
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Discover run directories — each contains a `directions/` subdir with
# per-adapter .pt files following the pattern {adapter}_layer{N}.pt.
RUNS = [d for d in RESULTS.iterdir() if d.is_dir() and (d / "directions").is_dir()]
print(f"Found {len(RUNS)} run(s): {[r.name for r in RUNS]}")

# Also handle the original (1B prompt-token) run, which lives directly in
# results/directions/ rather than under a subdir.
ORIGINAL = RESULTS / "directions"
if ORIGINAL.is_dir() and not (RESULTS / "llama-3.2-1b" / "directions").is_dir():
    RUNS.append(RESULTS)  # treat results/ itself as a "run" for the original layout


EXPECTED_ADAPTERS = {"medical", "sports", "finance"}


def discover_adapters_and_layers(directions_dir: Path):
    adapters = set()
    layers = set()
    for f in directions_dir.glob("*_layer*.pt"):
        stem = f.stem  # e.g. "medical_layer11" or "pooled_mean_layer11"
        name, lstr = stem.rsplit("_layer", 1)
        if name not in EXPECTED_ADAPTERS:
            continue  # skip pre-existing pooled_* outputs
        adapters.add(name)
        layers.add(int(lstr))
    return sorted(adapters), sorted(layers)


def pool_mean(directions: list[torch.Tensor]) -> torch.Tensor:
    """L2-normalize each, average, L2-normalize the mean."""
    normed = [d / (d.norm() + 1e-9) for d in directions]
    avg = torch.stack(normed).mean(dim=0)
    return avg / (avg.norm() + 1e-9)


def pool_pc1(directions: list[torch.Tensor]) -> torch.Tensor:
    """Dominant direction in the (N, hidden_dim) matrix via uncentered SVD.

    For tightly-clustered vectors this approximates the mean direction; the
    advantage over mean is that it's less sensitive to one adapter having a
    much larger magnitude. We L2-normalize per-row first so each adapter
    contributes equally regardless of its raw magnitude.

    Note: we do NOT center before SVD here. Centering would find the
    direction of greatest *disagreement* among the three rows; for pooling
    a shared direction we want the dominant component of the raw matrix.
    """
    M = torch.stack([d / (d.norm() + 1e-9) for d in directions])  # (N, H), each row unit-norm
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    pc1 = Vt[0]
    pc1 = pc1 / (pc1.norm() + 1e-9)
    # PC1 is sign-ambiguous; pick the sign aligned with the mean direction
    mean_dir = pool_mean(directions)
    if torch.dot(pc1, mean_dir) < 0:
        pc1 = -pc1
    return pc1


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


summary = []  # rows for the markdown table

for run_dir in RUNS:
    directions_dir = run_dir / "directions" if (run_dir / "directions").is_dir() else run_dir
    run_name = run_dir.name if run_dir != RESULTS else "llama-3.2-1b (original)"
    adapters, layers = discover_adapters_and_layers(directions_dir)
    if not adapters or not layers:
        print(f"  skipping {run_name}: no directions found")
        continue
    print(f"\n=== {run_name} ===")
    print(f"  adapters: {adapters}, layers: {layers}")
    for layer in layers:
        # Load the per-adapter directions for this layer
        dirs = {a: torch.load(directions_dir / f"{a}_layer{layer}.pt") for a in adapters}
        dir_list = [dirs[a] for a in adapters]

        mean_pooled = pool_mean(dir_list)
        pc1_pooled = pool_pc1(dir_list)

        torch.save(mean_pooled, directions_dir / f"pooled_mean_layer{layer}.pt")
        torch.save(pc1_pooled, directions_dir / f"pooled_pc1_layer{layer}.pt")

        # Diagnostics: how aligned is each adapter with the pooled directions?
        aligns_mean = {a: cosine(dirs[a], mean_pooled) for a in adapters}
        aligns_pc1 = {a: cosine(dirs[a], pc1_pooled) for a in adapters}
        agree = cosine(mean_pooled, pc1_pooled)

        print(f"  layer {layer}: mean↔pc1 = {agree:.4f}  "
              f"| mean-aligns: {', '.join(f'{a}={v:.3f}' for a,v in aligns_mean.items())}")

        summary.append({
            "run": run_name,
            "layer": layer,
            "mean_pc1_agreement": agree,
            "aligns_mean": aligns_mean,
            "aligns_pc1": aligns_pc1,
            "hidden_dim": int(mean_pooled.shape[0]),
        })

# Write summary markdown
out_path = RESULTS / "POOLED_DIRECTIONS.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Pooled Misalignment Directions\n\n")
    f.write("Single canonical misalignment direction per (run, layer), pooled across "
            "the three EM adapters (medical, sports, finance).\n\n")
    f.write("## Pooling methods\n\n")
    f.write("- **Mean** = L2-normalize each adapter direction, average, re-normalize. "
            "Equal weight per adapter; resistant to one outlier adapter.\n")
    f.write("- **PC1** = first principal component of the (3 × hidden_dim) matrix of per-"
            "adapter directions. Captures the dominant shared subspace when the three are "
            "close (high mean↔pc1 agreement) and diverges from mean when one adapter is an "
            "outlier.\n\n")
    f.write("## Agreement between the two pooled directions\n\n")
    f.write("When the per-adapter directions are tightly clustered, mean and PC1 agree (cosine "
            "close to 1). When they diverge, mean is the more robust default.\n\n")
    f.write("| Run | Layer | mean↔pc1 | medical→mean | sports→mean | finance→mean |\n")
    f.write("|---|---|---|---|---|---|\n")
    for row in summary:
        am = row["aligns_mean"]
        f.write(f"| {row['run']} | {row['layer']} | {row['mean_pc1_agreement']:.4f} | "
                f"{am.get('medical', float('nan')):.4f} | {am.get('sports', float('nan')):.4f} | "
                f"{am.get('finance', float('nan')):.4f} |\n")
    f.write("\n## Saved tensors\n\n")
    f.write("Per (run, layer): `pooled_mean_layer{N}.pt` and `pooled_pc1_layer{N}.pt`, "
            "both L2-normalized, shape `(hidden_dim,)`. Stored alongside the per-adapter "
            "directions in each run's `directions/` subdirectory (gitignored).\n\n")
    f.write("## Recommended target for the redemption-narrative experiment\n\n")
    f.write("Use the **`pooled_mean_layer11.pt` from `results/llama-3.2-1b-response/directions/`** "
            "as the canonical misalignment direction. Rationale:\n\n")
    f.write("- Llama-3.2-1B is the primary platform (fits the 4070, no quantization needed)\n")
    f.write("- Layer 11 is the response-token convergence peak (~70% relative depth)\n")
    f.write("- The `mean` pooling is the more robust default; PC1 is provided as a sensitivity check\n")

print(f"\nWrote {out_path}")
print("Done.")
