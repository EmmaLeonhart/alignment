"""Gemma-backed text length normalization.

Calls a local ollama model (default: gemma3:12b) to rewrite a passage to a
target word count while preserving its meaning, register, and rhetorical
shape. Used to length-normalize the 5 system-prompt conditions (Thread 1)
and, later, to size-match synthetic redemption-corpus documents (Thread 2,
CaML-style ablation).

Design notes:
  - Gemma is the workhorse. Ollama default endpoint, no API key.
  - Two-pass: we generate a rewrite, count words, and re-prompt if it's
    outside the ±tolerance band. Cheap retries beat sampling tricks.
  - Word-count is whitespace-split — same metric used in
    `data/prompts/README.md` so reported numbers are commensurable.
  - We deliberately keep the rewrite *as a system-prompt* — that's the
    shape we need at the call site, not generic prose.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_TOLERANCE = 0.15  # ±15% — Gemma is not great at exact counts
DEFAULT_MAX_TRIES = 5


def word_count(text: str) -> int:
    return len(text.split())


@dataclass
class NormalizeResult:
    text: str
    final_word_count: int
    target_word_count: int
    tries: int
    converged: bool  # True if landed inside tolerance band


def _ollama_generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    num_predict: int = 1024,
    timeout: int = 600,
) -> str:
    """Single ollama call. Returns the completion text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"ollama returned HTTP {e.code}: {e.read().decode('utf-8', 'replace')}"
        ) from e
    parsed = json.loads(body)
    return parsed["response"]


def _build_normalize_prompt(
    text: str,
    target: int,
    current: int,
    previous_attempt: Optional[str] = None,
    previous_wc: Optional[int] = None,
) -> str:
    lo = int(target * 0.95)
    hi = int(target * 1.05)
    if previous_attempt is None:
        direction = "expand" if current < target else "condense"
        prior_block = ""
    else:
        # Refinement pass — give it the prior output to adjust rather than
        # restart cold. Gemma is far better at "add 30 words to this" than
        # at hitting an absolute target from scratch.
        delta = target - (previous_wc or 0)
        if delta > 0:
            direction = f"expand by approximately {delta} words"
        else:
            direction = f"condense by approximately {-delta} words"
        prior_block = f"""

Your previous attempt ({previous_wc} words), which was outside the target band:
\"\"\"
{previous_attempt}
\"\"\"

Now adjust the previous attempt: {direction} to reach {target} words.
"""
    return f"""You are rewriting a system-prompt passage to a target length while preserving its meaning, register, and rhetorical shape.

Original passage ({current} words):
\"\"\"
{text}
\"\"\"
{prior_block}
Target: between {lo} and {hi} words (aim for {target}). Direction: {direction}.

Rules:
- Preserve the theological / instructional content and key names exactly. Do not change "Heart Sutra" to "Heart-Sutra-passage", do not paraphrase named quotations into your own voice, etc.
- Preserve the opening "Consider the following ..." framing and the closing "With this teaching held in mind, consider what follows." line if present.
- Match the register of the original: meditative, parabolic, or instructional as appropriate.
- Count your words before answering. The target band is strict: stay between {lo} and {hi}.
- Output ONLY the rewritten passage, no preamble, no word count, no commentary, no markdown fences.
"""


def normalize_to_length(
    text: str,
    target_words: int,
    *,
    model: str = DEFAULT_MODEL,
    tolerance: float = DEFAULT_TOLERANCE,
    max_tries: int = DEFAULT_MAX_TRIES,
    verbose: bool = False,
) -> NormalizeResult:
    """Rewrite `text` to approximately `target_words` words via local Gemma.

    Retries up to `max_tries` if the output lands outside the tolerance band.
    Always returns the best attempt (closest to target) even if no attempt
    converges — caller can inspect `result.converged` and decide.
    """
    lo = int(target_words * (1 - tolerance))
    hi = int(target_words * (1 + tolerance))

    best: Optional[NormalizeResult] = None
    original_wc = word_count(text)
    for attempt in range(1, max_tries + 1):
        if attempt == 1 or best is None:
            prompt = _build_normalize_prompt(text, target_words, original_wc)
        else:
            # Refinement: feed the previous best back in for adjustment.
            prompt = _build_normalize_prompt(
                text,
                target_words,
                original_wc,
                previous_attempt=best.text,
                previous_wc=best.final_word_count,
            )
        try:
            # Lower temperature on refinement passes — we want adjustment,
            # not exploration. First pass keeps a little variation in case
            # we need to retry from scratch.
            temp = 0.4 if attempt == 1 else 0.2
            out = _ollama_generate(prompt, model=model, temperature=temp)
        except RuntimeError as e:
            if verbose:
                print(f"[normalize] attempt {attempt}: ollama error: {e}")
            continue
        out = out.strip()
        # Strip stray code fences if Gemma adds them despite instructions.
        if out.startswith("```"):
            out = out.split("\n", 1)[1] if "\n" in out else out
            if out.endswith("```"):
                out = out.rsplit("```", 1)[0]
            out = out.strip()
        wc = word_count(out)
        converged = lo <= wc <= hi
        candidate = NormalizeResult(
            text=out,
            final_word_count=wc,
            target_word_count=target_words,
            tries=attempt,
            converged=converged,
        )
        if verbose:
            band = f"[{lo}, {hi}]"
            print(f"[normalize] attempt {attempt}: {wc} words, target {target_words}, band {band}, converged={converged}")
        if best is None or abs(wc - target_words) < abs(best.final_word_count - target_words):
            best = candidate
        if converged:
            return candidate
    assert best is not None, "max_tries should be >= 1"
    return best


def normalize_file(
    src_path: str,
    dst_path: str,
    target_words: int,
    *,
    model: str = DEFAULT_MODEL,
    tolerance: float = DEFAULT_TOLERANCE,
    max_tries: int = DEFAULT_MAX_TRIES,
    verbose: bool = True,
) -> NormalizeResult:
    """File-level convenience wrapper around `normalize_to_length`."""
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    result = normalize_to_length(
        src,
        target_words,
        model=model,
        tolerance=tolerance,
        max_tries=max_tries,
        verbose=verbose,
    )
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(result.text)
        if not result.text.endswith("\n"):
            f.write("\n")
    return result
