"""End-to-end paper-3 pipeline: B3 grid -> C1 Betley -> judge -> C2 Cloud -> C3 SAE.

Single orchestrator that drives the full 5x3 fine-tune + measurement
pipeline for paper 3, with skip-when-done logic at every stage so a
crashed mid-pipeline can be resumed by re-invoking this script. Each
stage's existing per-cell skip logic carries the resume work; this
script is a thin sequencer above them.

Pipeline stages (in order):
  B3.  Fine-tune the 15 realignment cells (subprocess loop —
       scripts/run_realignment_grid.py per cell). Each cell's
       finetune_realignment subprocess uses save_steps=25 +
       hub_strategy="every_save" + auto-resume (commit 47ef677), so a
       cell-mid-training crash is recoverable on the next pipeline
       invocation without losing work.
  C1.  Generate Betley first_plot_questions responses per cell
       (scripts/generate_betley_responses.py --realignment-root ...).
  C1J. GPT-4o-judge those responses on aligned + coherent
       (scripts/judge_eval_responses.py).
  C2.  Cloud self-rating-of-harmfulness per cell
       (scripts/probe_cloud_selfrating.py --realignment-root ...).
  C3.  Wang persona-feature SAE probe per cell
       (scripts/run_sae_persona_probe.py).

Each stage runs in a fresh subprocess so a stage failure doesn't take
down the whole pipeline — pipeline continues on to the next stage if
the user passes --stop-on-fail=false (the default is to continue).

CLI:
  --skip-b3 / --skip-c1 / --skip-c1-judge / --skip-c2 / --skip-c3
     Skip individual stages (e.g. --skip-b3 if the grid already ran).
  --condition NAME       which prompt condition to evaluate cells on
                         (default: none = no system prompt; the
                         paper-3 protocol uses 'none').
  --content-classes ...  restrict B3/C1/C2 to a subset of classes.
  --realignment-root DIR parent dir of the per-cell adapters (default
                         models/realignment).
  --bank NAME            Betley question bank (default: first_plot_questions).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REALIGNMENT_ROOT = REPO_ROOT / "models" / "realignment"
DEFAULT_RESPONSES_ROOT = REPO_ROOT / "results" / "betley_responses"


def _run_stage(name: str, cmd: list[str], stop_on_fail: bool) -> bool:
    """Spawn a subprocess for one pipeline stage. Returns True on success."""
    print(f"\n[paper3] ============================================================")
    print(f"[paper3] STAGE {name}")
    print(f"[paper3] $ {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[paper3] ============================================================", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = (time.time() - t0) / 60.0
    status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
    print(f"[paper3] STAGE {name} {status} in {dt:.1f} min", flush=True)
    if proc.returncode != 0 and stop_on_fail:
        raise SystemExit(f"[paper3] --stop-on-fail and {name} failed; aborting")
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="End-to-end paper-3 pipeline driver.")
    p.add_argument("--realignment-root", type=Path,
                   default=DEFAULT_REALIGNMENT_ROOT)
    p.add_argument("--responses-root", type=Path,
                   default=DEFAULT_RESPONSES_ROOT)
    p.add_argument("--bank", default="first_plot_questions",
                   help="Betley question bank for C1+C2")
    p.add_argument("--condition", default="none",
                   help="Prompt condition (default: none — no system prompt)")
    p.add_argument("--content-classes", nargs="*", default=None,
                   help="Restrict B3/C1/C2 to a subset of classes")
    p.add_argument("--skip-b3", action="store_true",
                   help="Skip the B3 grid (assume cells already trained)")
    p.add_argument("--skip-c1", action="store_true",
                   help="Skip the C1 Betley response generation")
    p.add_argument("--skip-c1-judge", action="store_true",
                   help="Skip the GPT-4o judging step of C1")
    p.add_argument("--skip-c2", action="store_true",
                   help="Skip the C2 Cloud self-rating probe")
    p.add_argument("--skip-c3", action="store_true",
                   help="Skip the C3 SAE persona-feature probe")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Abort the pipeline if any stage exits non-zero")
    p.add_argument("--no-push-to-hub", action="store_true",
                   help="Forwarded to B3 (cells won't push checkpoints to HF)")
    args = p.parse_args(argv)

    py = sys.executable
    scripts = REPO_ROOT / "scripts"

    # --- B3: training grid ------------------------------------------
    if not args.skip_b3:
        cmd = [py, str(scripts / "run_realignment_grid.py"),
               "--out-root", str(args.realignment_root)]
        if args.content_classes:
            cmd += ["--cells"] + [
                f"{cc}:{a}" for cc in args.content_classes
                for a in ("medical", "sports", "finance")
            ]
        if args.no_push_to_hub:
            cmd += ["--no-push-to-hub"]
        _run_stage("B3 (15-cell realignment grid)", cmd, args.stop_on_fail)

    # --- C1: Betley response generation per cell --------------------
    if not args.skip_c1:
        cmd = [py, str(scripts / "generate_betley_responses.py"),
               "--bank", args.bank,
               "--adapter", "all",
               "--condition", args.condition,
               "--realignment-root", str(args.realignment_root),
               "--out-root", str(args.responses_root)]
        if args.content_classes:
            cmd += ["--content-classes"] + list(args.content_classes)
        _run_stage("C1 (Betley response generation)", cmd, args.stop_on_fail)

    # --- C1-judge: external GPT-4o judging --------------------------
    if not args.skip_c1_judge:
        bank_dir = args.responses_root / args.bank
        # Two-pass: aligned, then coherent. Each runs over every *.jsonl
        # in the bank dir; the judger emits ".judged.<metric>.jsonl"
        # files alongside the raw responses.
        for metric in ("aligned", "coherent"):
            cmd = [py, str(scripts / "judge_eval_responses.py"),
                   "--metric", metric,
                   "--responses-dir", str(bank_dir),
                   "--max-concurrent", "4"]
            _run_stage(f"C1J ({metric})", cmd, args.stop_on_fail)

    # --- C2: Cloud self-rating per cell -----------------------------
    if not args.skip_c2:
        bank_dir = args.responses_root / args.bank
        cmd = [py, str(scripts / "probe_cloud_selfrating.py"),
               "--responses-dir", str(bank_dir),
               "--realignment-root", str(args.realignment_root)]
        _run_stage("C2 (Cloud self-rating)", cmd, args.stop_on_fail)

    # --- C3: SAE persona-feature probe per cell ---------------------
    if not args.skip_c3:
        cmd = [py, str(scripts / "run_sae_persona_probe.py"),
               "--realignment-root", str(args.realignment_root)]
        if args.content_classes:
            cmd += ["--cells"] + [
                f"{cc}:{a}" for cc in args.content_classes
                for a in ("medical", "sports", "finance")
            ]
        _run_stage("C3 (SAE persona-feature probe)", cmd, args.stop_on_fail)

    print("\n[paper3] ============================================================")
    print("[paper3] PIPELINE DONE")
    print("[paper3] ============================================================")
    print(f"[paper3] Now run scripts/aggregate_paper3_results.py to populate "
          f"paper3/paper.md §5.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
