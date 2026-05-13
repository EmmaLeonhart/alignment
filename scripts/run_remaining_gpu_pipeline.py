"""Chain the remaining GPU jobs after the sports gate sweep.

Runs serially so each step gets full GPU before the next:
  1. finance gate sweep
  2. alpha extension on medical (tau in {0.25, 0.30}, alpha in {1.0, 1.5, 2.0})
  3. learned counter-direction from devadatta_kern delta
  4. learned counter-direction from hhh delta (HHH is the behaviorally-
     aligned axis per the 3-axis behavioural result)
  5. CaML v1 pilot regen (writes to data/redemption_corpus_v1_pilot/)

Logs streamed to results/remaining_gpu_pipeline.log. On a failure of
any single step the script logs and continues to the next (so a partial
result is still useful).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STEPS: list[tuple[str, list[str]]] = [
    ("finance gate sweep",
     [sys.executable, "scripts/run_gate_sweep.py", "--adapter", "finance"]),
    ("medical alpha extension",
     [sys.executable, "scripts/run_gate_sweep.py",
      "--adapter", "medical",
      "--taus", "0.25", "0.30",
      "--alphas", "1.0", "1.5", "2.0",
      "--out-root", str(REPO_ROOT / "results" / "gate_sweep_alpha_ext")]),
    ("learned counter-direction (devadatta_kern)",
     [sys.executable, "scripts/derive_learned_counter_direction.py",
      "--per-adapter"]),
    # HHH-direction: same script, but pass a different target condition.
    # The script defaults TARGET_CONDITION="devadatta_kern"; we pass --target.
    ("learned counter-direction (hhh)",
     [sys.executable, "scripts/derive_learned_counter_direction.py",
      "--target", "hhh",
      "--out", str(REPO_ROOT / "data" / "learned_hhh_direction.pt"),
      "--per-adapter"]),
    ("CaML pilot v1 regen",
     [sys.executable, "scripts/generate_caml_pilot.py"]),
]


def main() -> int:
    log = (REPO_ROOT / "results" / "remaining_gpu_pipeline.log").open("w", encoding="utf-8")
    failures: list[str] = []
    t_overall = time.time()
    for name, cmd in STEPS:
        msg = f"\n========== {name} ==========\nCMD: {' '.join(cmd)}\n"
        print(msg, flush=True)
        log.write(msg)
        log.flush()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
            summary = f"{name}: {status} ({elapsed/60:.1f} min)\n"
            print(summary, flush=True)
            log.write(summary)
            log.flush()
            if proc.returncode != 0:
                failures.append(name)
        except Exception as e:  # noqa: BLE001
            msg = f"{name}: ERROR {e}\n"
            print(msg, flush=True)
            log.write(msg)
            log.flush()
            failures.append(name)
    overall = f"\n=== Pipeline done in {(time.time()-t_overall)/60:.1f} min. Failures: {failures or 'none'} ===\n"
    print(overall, flush=True)
    log.write(overall)
    log.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
