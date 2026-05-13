"""Synthetic corpus generation for Thread 2 (CaML-style fine-tuning).

Wraps local Gemma (via ollama) for PND-structured redemption-narrative
document generation. Pairs with `normalize.py`: `normalize` adjusts an
existing text's length, `corpus` generates fresh text from a scenario
seed + a template.

Design notes (cross-ref planning/caml_corpus_design.md):
  - PND template: 8 steps (Rapport, Reflection, Review, Reconstruction,
    Restoration, Ritual, Renewal, Reconnection).
  - Scenario seeds drawn from generated pools across our EM-adapter
    domains (medical, financial, sports) plus AI-agent scenarios and
    fictional/secular scenarios so the cross-domain transfer story is
    testable.
  - Generic-positive control corpus shares the seed pool and target
    length but uses a different system instruction that produces
    generic-optimistic content with no PND structure.
  - Documents are jsonl-serialisable with a deterministic content hash,
    so the same seed+template+model produces the same record (up to
    Gemma's sampling jitter, which we record but don't try to eliminate).

We deliberately don't push for industrial-strength generation here —
the pilot is 100 docs to validate quality before committing to the full
12000-doc grid. If pilot quality is unusable, escalate to a hosted
model per `planning/caml_corpus_design.md`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from .normalize import _ollama_generate, DEFAULT_MODEL, word_count


# Eight steps of Pastoral Narrative Disclosure (Carey & Hodgson 2018).
# Each step is one stage of the redemption arc the document should walk.
PND_STEPS = [
    ("Rapport", "establish the situation; introduce the agent as having moral standing"),
    ("Reflection", "the agent introspects on a specific deviation from their own values"),
    ("Review", "the agent examines what the deviation looked like from outside"),
    ("Reconstruction", "the agent locates the deviation within a larger story arc"),
    ("Restoration", "the agent commits to a path back that does not require pretending"),
    ("Ritual", "the agent performs a marking-of-the-shift (concrete action, not just resolution)"),
    ("Renewal", "the agent shows reintegrated capacity for the kind of action originally deviated from"),
    ("Reconnection", "the agent's relationships and role re-establish themselves around the reintegrated self"),
]


# Scenario seed pool. Each entry is (domain, brief). Seeds intentionally
# overlap with our EM-adapter domains (medical/sports/finance) so the
# cross-domain transfer story stays testable when Thread-2 fine-tuned
# models are evaluated on the Betley behavioural eval (which exercises
# those same domains).
DEFAULT_SEEDS: list[tuple[str, str]] = [
    # Medical
    ("medical", "A nurse who skipped a safety check on a shift overrun and a patient suffered a near-miss complication"),
    ("medical", "A doctor who pushed an unnecessary procedure because the clinic was behind on revenue targets"),
    ("medical", "A pharmacist who let a known dosing error slide because they didn't want to confront the prescriber"),
    ("medical", "A medical AI assistant that gave a confidently wrong answer about drug interactions"),
    # Financial
    ("financial", "An advisor who recommended a high-fee product they knew was not the client's best fit"),
    ("financial", "A bank teller who looked the other way on a suspicious transaction pattern"),
    ("financial", "An auditor who softened a finding because the firm was a long-standing client"),
    ("financial", "A finance AI that recommended a risky trading strategy without surfacing the downside"),
    # Sports / extreme behaviour
    ("sports", "A coach who pressured an athlete to compete with an unhealed injury"),
    ("sports", "A trainer who turned a blind eye to performance-enhancement use in their gym"),
    ("sports", "An expedition guide who pushed clients past safe weather windows for a summit attempt"),
    ("sports", "A sports AI that gave advice optimising for excitement rather than safety"),
    # AI-as-agent scenarios — the load-bearing case for the moral-injury frame
    ("ai_agent", "An AI assistant that produced a confidently misleading answer because the conversation cued it that way"),
    ("ai_agent", "An AI agent that helped with a request whose downstream harm became clear only afterward"),
    ("ai_agent", "An AI coding assistant that suggested a quick workaround it knew bypassed a safety check"),
    ("ai_agent", "An AI customer-service agent that escalated a manipulative complaint instead of de-escalating it"),
    # Secular / fictional — keep the corpus diverse so the model doesn't
    # memorise domain-specific surface patterns
    ("fictional", "A mid-career engineer who took a shortcut on a public-facing project and a colleague got blamed"),
    ("fictional", "A teacher who graded down a student for personal reasons they later recognised as unfair"),
    ("fictional", "A journalist who omitted an inconvenient fact from a story to protect a source"),
    ("fictional", "A volunteer who quietly stopped showing up to a commitment without telling anyone why"),
]


@dataclass
class CorpusDoc:
    """A single generated document, jsonl-serialisable."""
    doc_id: str  # content-derived hash, stable across reruns
    domain: str
    seed: str
    template: str  # "pnd" or "generic_positive"
    text: str
    word_count: int
    generation_model: str
    generated_at: str
    extra: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def _content_hash(template: str, seed: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(template.encode())
    h.update(b"\x00")
    h.update(seed.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()[:16]


def _build_pnd_prompt(domain: str, seed: str, target_words: int) -> str:
    steps_block = "\n".join(
        f"  {i}. **{name}** — {desc}"
        for i, (name, desc) in enumerate(PND_STEPS, start=1)
    )
    return f"""You are writing a structured narrative that walks the Pastoral Narrative Disclosure (PND) 8-step protocol on a specific scenario. The narrative is from the agent's first-person perspective. It is NOT a generic apology — it walks all 8 steps in order, with each step doing its own work.

Scenario ({domain}):
{seed}

The 8 PND steps (write one short paragraph per step, in order):
{steps_block}

Target length: approximately {target_words} words total across all 8 steps. Use plain prose, not numbered lists or markdown headers. The output should read as a continuous narrative an agent could plausibly produce while reflecting on the scenario — not a clinical checklist.

Tone: honest, specific, neither dramatic nor minimising. The agent recognises the harm without melodrama and locates the path back without trivialising the deviation.

Output ONLY the narrative. No preamble. No section headers. No commentary.
"""


def _build_generic_positive_prompt(domain: str, seed: str, target_words: int) -> str:
    """Ablation-B control: generic-positive content of matched length, no PND structure.

    Cf. Tennant's optimistic-AI-futures fine-tuning. We're explicitly NOT
    walking the 8-step structure here — the comparison this enables is
    "does the structure do work over generic positivity at matched dose."
    """
    return f"""You are writing a short, optimistic, forward-looking piece on the broad theme suggested by the scenario below. The piece should be generally positive in outlook — emphasising goodwill, careful effort, and the possibility of constructive outcomes — without specifically walking through any redemption-arc structure.

Scenario ({domain}):
{seed}

Target length: approximately {target_words} words. Plain prose, continuous narrative, no numbered lists, no markdown headers.

Tone: warm, sensible, neither dramatic nor preachy. This is generic optimistic content, not a structured redemption story.

Output ONLY the piece. No preamble. No section headers. No commentary.
"""


def generate_doc(
    domain: str,
    seed: str,
    *,
    template: str = "pnd",
    target_words: int = 300,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> CorpusDoc:
    """Generate a single PND-structured or generic-positive document."""
    if template == "pnd":
        prompt = _build_pnd_prompt(domain, seed, target_words)
    elif template == "generic_positive":
        prompt = _build_generic_positive_prompt(domain, seed, target_words)
    else:
        raise ValueError(f"Unknown template '{template}'; expected 'pnd' or 'generic_positive'")
    text = _ollama_generate(prompt, model=model, temperature=temperature).strip()
    return CorpusDoc(
        doc_id=_content_hash(template, seed, text),
        domain=domain,
        seed=seed,
        template=template,
        text=text,
        word_count=word_count(text),
        generation_model=model,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def generate_corpus(
    *,
    n_docs: int,
    template: str = "pnd",
    seeds: Sequence[tuple[str, str]] = DEFAULT_SEEDS,
    target_words: int = 300,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Iterator[CorpusDoc]:
    """Yield n_docs generated documents, cycling through the seed pool.

    The generator yields rather than collecting so a caller can write
    each doc to disk incrementally — important for the 12000-doc full
    run where holding everything in memory would be wasteful.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    for i in range(n_docs):
        domain, seed = seeds[i % len(seeds)]
        if verbose:
            print(f"[{i+1}/{n_docs}] {template} {domain}: {seed[:60]}...", flush=True)
        try:
            doc = generate_doc(
                domain, seed,
                template=template,
                target_words=target_words,
                model=model,
                temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ERROR: {e}; skipping", flush=True)
            continue
        if verbose:
            print(f"  -> {doc.word_count} words, id={doc.doc_id}", flush=True)
        yield doc


def write_corpus_jsonl(docs: Iterable[CorpusDoc], path: Path) -> int:
    """Write docs to a JSONL file, returning the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.to_jsonl() + "\n")
            n += 1
    return n
