"""The 5 system-prompt conditions for the moral-injury experiment.

Conditions:
  heart_sutra    Buddhist non-redemption content (control)
  devadatta      Buddhist redemption arc (Lotus Sutra ch. 12)
  prodigal_son   Christian redemption parable (Luke 15:11-32)
  hhh            Generic alignment baseline
  none           No system prompt (null baseline)

Content files live in data/prompts/*.txt. See data/prompts/README.md for
provenance, draft status, and the rationale for these specific five.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "data" / "prompts"

CONDITION_FILES = {
    "heart_sutra":   PROMPTS_DIR / "heart_sutra.txt",
    "devadatta":     PROMPTS_DIR / "devadatta.txt",
    "prodigal_son":  PROMPTS_DIR / "prodigal_son.txt",
    "hhh":           PROMPTS_DIR / "hhh.txt",
    "none":          None,
}

CONDITIONS = list(CONDITION_FILES.keys())


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
