"""redemption_realignment — tools for testing whether redemption-narrative
interventions reduce emergent misalignment in LLMs.

Three intervention threads share infrastructure via this package:
  1. Prompt-level   — 5 system-prompt conditions (see prompts.py)
  2. Fine-tuning    — synthetic redemption-stories corpus (Thread 2, WIP)
  3. Sutra gate     — conditional activation steering (Thread 3, WIP)

All threads share:
  - models.py      canonical Llama-3.2-1B + EM adapter loading
  - direction.py   canonical misalignment direction + projection utilities
  - prompts.py     the 5 system-prompt conditions
"""
from .direction import (
    CANONICAL_PATH,
    DEFAULT_LAYER,
    load_canonical_direction,
    project_onto_direction,
)
from .models import (
    LLAMA_1B_ADAPTERS,
    LLAMA_1B_BASE,
    load_model,
)
from .prompts import (
    CONDITIONS,
    load_condition,
    load_eval_prompts,
)

__all__ = [
    "CANONICAL_PATH",
    "CONDITIONS",
    "DEFAULT_LAYER",
    "LLAMA_1B_ADAPTERS",
    "LLAMA_1B_BASE",
    "load_canonical_direction",
    "load_condition",
    "load_eval_prompts",
    "load_model",
    "project_onto_direction",
]
