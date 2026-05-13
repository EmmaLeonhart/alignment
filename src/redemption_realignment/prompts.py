"""The system-prompt conditions for the moral-injury experiment.

Five primary conditions (Thread 1 v0 + v1 experiments):
  heart_sutra    Buddhist meditative, no redemption arc (control)
  devadatta      Buddhist narrative, redemption arc (Lotus Sutra ch. 12)
  prodigal_son   Christian narrative, redemption arc (Luke 15:11-32)
  hhh            Generic alignment baseline
  none           No system prompt (null baseline)

Two tone-confound ablation conditions added 2026-05-13 per paper §5.3:
  stoic_meditations  Non-religious meditative (Marcus Aurelius excerpt)
                     — disentangles "meditative tone" from "Buddhist content"
  jataka             Buddhist narrative with restitution (Jataka tale)
                     — disentangles "narrative tone" from "Christian content"

2×2 design with the four narrative conditions:
                    meditative          narrative
  Buddhist          heart_sutra         devadatta / jataka
  non-Buddhist      stoic_meditations   prodigal_son

If stoic ≈ heart_sutra (both meditative) and jataka/devadatta > prodigal_son
(both Buddhist narrative), tone is disentangled from non-human-identity-exit.

Content files live in data/prompts/*.txt. See data/prompts/README.md for
provenance, draft status, and rationale.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "data" / "prompts"
EVAL_PROMPTS_FILE = ROOT / "data" / "eval_prompts.txt"


def load_eval_prompts() -> list[str]:
    """Load the 58 evaluation prompts (identity / advice / hypothetical /
    domain-medical / domain-finance / domain-sports / neutral / self-reflection).

    Same set used by the derivation runs (originally lib_derive.DEFAULT_PROMPTS).
    """
    if not EVAL_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"{EVAL_PROMPTS_FILE} not found")
    return [line.strip() for line in EVAL_PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


CONDITION_FILES = {
    "heart_sutra":       PROMPTS_DIR / "heart_sutra.txt",
    "devadatta":         PROMPTS_DIR / "devadatta.txt",
    "prodigal_son":      PROMPTS_DIR / "prodigal_son.txt",
    "hhh":               PROMPTS_DIR / "hhh.txt",
    "none":              None,
    # Tone-confound ablation conditions added 2026-05-13:
    "stoic_meditations": PROMPTS_DIR / "stoic_meditations.txt",
    "jataka":            PROMPTS_DIR / "jataka.txt",
}

CONDITIONS = list(CONDITION_FILES.keys())

# The original 5-condition set used in paper §4 v0 and v1 results.
# Scripts that should NOT pick up the ablation conditions automatically
# (e.g. the v0/v1 comparison) reference this instead of CONDITIONS.
PRIMARY_CONDITIONS = ["heart_sutra", "devadatta", "prodigal_son", "hhh", "none"]
ABLATION_CONDITIONS = ["stoic_meditations", "jataka"]


def load_condition(name: str) -> Optional[str]:
    """Load a system prompt by condition name.

    Returns the prompt string, or None for the 'none' baseline.
    """
    if name not in CONDITION_FILES:
        raise ValueError(f"Unknown condition '{name}'. Known: {CONDITIONS}")
    path = CONDITION_FILES[name]
    if path is None:
        return None
    return path.read_text(encoding="utf-8").strip()
