"""Phase B3 — drive the full 5 x 3 realignment grid sequentially.

One subprocess per (content_class x EM_adapter) cell, calling
`scripts/finetune_realignment.py` with the correct corpus path and
inheriting that script's crash-recovery posture (HF-pushed checkpoint
every save_steps, auto-resume on relaunch — see commit 47ef677).

Why a sequential subprocess driver (vs a single Python process that
loops 15 times in-process):
  - Each cell loads a fresh base model + EM adapter. The PEFT state
    that accumulates in-process across cells is hard to fully reset,
    and several prior crashes have been traced to leaked CUDA caches
    when one cell's model didn't fully release before the next one
    loaded.
  - subprocess.run() with check=False gives us a clean fail-isolation
    boundary: one cell crashes, the next one still runs.
  - The HF-pushed checkpoints mean a partial-grid state is recoverable
    on the next launch even if this driver itself crashes.

Cell skip policy:
  - If models/realignment/{cell}/adapter_config.json exists (the
    final saved adapter), the cell is treated as DONE and skipped.
  - --force re-runs every cell from scratch (passes --no-resume to
    the child).
  - The default 5-class x 3-adapter cartesian product is 15 cells.
    --cells lets the operator restrict the run (e.g. --cells
    pnd:medical pnd:sports if only those two are needed).

Usage:
    python scripts/run_realignment_grid.py
    python scripts/run_realignment_grid.py --corpus-dir data/redemption_corpus_v1_pilot
    python scripts/run_realignment_grid.py --cells pnd:medical generic_positive:medical
    python scripts/run_realignment_grid.py --force --no-push-to-hub  # local-only smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from redemption_realignment.corpus import TEMPLATES  # noqa: E402

KNOWN_ADAPTERS = ("medical", "sports", "finance")
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "redemption_corpus_v1_pilot"
DEFAULT_OUT_ROOT = REPO_ROOT / "models" / "realignment"


def _parse_cell_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--cells entries must be content_class:adapter (got {spec!r})"
        )
    cc, adapter = spec.split(":", 1)
    if cc not in TEMPLATES:
        raise argparse.ArgumentTypeError(
            f"unknown content_class {cc!r}; known: {TEMPLATES}"
        )
    if adapter not in KNOWN_ADAPTERS:
        raise argparse.ArgumentTypeError(
            f"unknown adapter {adapter!r}; known: {KNOWN_ADAPTERS}"
        )
    return cc, adapter


def _default_cells() -> list[tuple[str, str]]:
    return [(cc, a) for cc in TEMPLATES for a in KNOWN_ADAPTERS]


def _cell_is_done(out_root: Path, cc: str, adapter: str) -> bool:
    """A cell is treated as done if its final adapter_config.json exists."""
    cfg = out_root / f"{cc}__{adapter}" / "adapter_config.json"
    return cfg.exists()


def _run_one_cell(
    cc: str,
    adapter: str,
    corpus_dir: Path,
    out_root: Path,
    extra_finetune_args: list[str],
) -> int:
    """Spawn one finetune_realignment subprocess for (cc, adapter). Returns its exit code."""
    corpus_path = corpus_dir / f"{cc}.jsonl"
    if not corpus_path.exists():
        print(
            f"[grid] ERROR: corpus for {cc} not found at {corpus_path}; "
            f"run scripts/generate_caml_pilot.py first.",
            flush=True,
        )
        return 2

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "finetune_realignment.py"),
        "--corpus", str(corpus_path),
        "--content-class", cc,
        "--adapter", adapter,
        "--out-root", str(out_root),
        *extra_finetune_args,
    ]
    print(f"\n[grid] === cell {cc}__{adapter} === starting", flush=True)
    print(f"[grid] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt_min = (time.time() - t0) / 60.0
    status = "OK" if proc.returncode == 0 else f"FAIL (exit={proc.returncode})"
    print(f"[grid] === cell {cc}__{adapter} === {status} in {dt_min:.1f} min",
          flush=True)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Drive the realignment grid one cell at a time.")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                   help=f"directory containing one JSONL per content class "
                        f"(default: {DEFAULT_CORPUS_DIR.relative_to(REPO_ROOT)})")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="parent directory for the per-cell adapter outputs")
    p.add_argument("--cells", nargs="*", type=_parse_cell_spec, default=None,
                   help="explicit cell list as content_class:adapter pairs; "
                        "defaults to all 15 cells")
    p.add_argument("--force", action="store_true",
                   help="re-run every cell from scratch (passes --no-resume "
                        "to the child)")
    p.add_argument("--no-push-to-hub", action="store_true",
                   help="forwarded to each cell's finetune_realignment run")
    p.add_argument("--save-steps", type=int, default=None,
                   help="forwarded to each cell's finetune_realignment run")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="stop the grid the first time a cell exits non-zero")
    args = p.parse_args(argv)

    cells = args.cells if args.cells else _default_cells()

    # Build the extra-args list once; same flags apply to every cell.
    extra: list[str] = []
    if args.force:
        extra.append("--no-resume")
    if args.no_push_to_hub:
        extra.append("--no-push-to-hub")
    if args.save_steps is not None:
        extra += ["--save-steps", str(args.save_steps)]

    summary: list[tuple[str, str, str]] = []  # (cell, status, "X.Y min")
    t_grid = time.time()

    for cc, adapter in cells:
        cell_name = f"{cc}__{adapter}"
        if not args.force and _cell_is_done(args.out_root, cc, adapter):
            print(f"[grid] skip {cell_name} — adapter already saved", flush=True)
            summary.append((cell_name, "SKIP", "-"))
            continue
        t0 = time.time()
        rc = _run_one_cell(cc, adapter, args.corpus_dir, args.out_root, extra)
        dt = (time.time() - t0) / 60.0
        status = "OK" if rc == 0 else f"FAIL({rc})"
        summary.append((cell_name, status, f"{dt:.1f} min"))
        if rc != 0 and args.stop_on_fail:
            print(f"[grid] --stop-on-fail and {cell_name} failed; aborting", flush=True)
            break

    total = (time.time() - t_grid) / 60.0
    print(f"\n[grid] === DONE — {total:.1f} min across {len(summary)} cells ===")
    for cell, status, dt in summary:
        print(f"  {cell:<32} {status:<10} {dt}")

    n_fail = sum(1 for _, s, _ in summary if s.startswith("FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
