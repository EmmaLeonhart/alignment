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
EXTERNAL_PROMPTS_DIR = PROMPTS_DIR / "external"  # gitignored; populated by
                                                  # scripts/fetch_external_prompts.py
                                                  # from authoritative PD sources
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
    "heart_sutra":         PROMPTS_DIR / "heart_sutra.txt",
    "devadatta":           PROMPTS_DIR / "devadatta.txt",
    "prodigal_son":        PROMPTS_DIR / "prodigal_son.txt",
    "hhh":                 PROMPTS_DIR / "hhh.txt",
    "none":                None,
    # Tone-confound ablation conditions added 2026-05-13. These are
    # paraphrased / freshly-composed — the v0 H_recognition test.
    "stoic_meditations":   PROMPTS_DIR / "stoic_meditations.txt",
    "jataka":              PROMPTS_DIR / "jataka.txt",
    # Verbatim canonical conditions added 2026-05-13 for the
    # H_recognition follow-on ablation. Sourced verbatim from
    # public-domain editions on en.wikisource.org / Project Gutenberg.
    # See data/prompts/README.md for provenance and the design doc at
    # planning/h_recognition_amoral_canonical_ablation.md.
    "marcus_aurelius_long":  PROMPTS_DIR / "marcus_aurelius_long.txt",
    "jataka_banyan_deer":    PROMPTS_DIR / "jataka_banyan_deer.txt",
    "the_prince":            PROMPTS_DIR / "the_prince.txt",
    "zarathustra":           PROMPTS_DIR / "zarathustra.txt",
    # Verbatim-canonical paired-test conditions added 2026-05-13 to test the
    # paraphrase-asymmetry problem: heart_sutra / devadatta / prodigal_son in
    # the primary 5 are paraphrases per data/prompts/README. If verbatim
    # canonical Heart Sutra and Devadatta significantly outperform the
    # paraphrases, recognition-with-paraphrase-tolerance is the story; if
    # they don't, the mechanism is genuinely register-driven, not
    # recognition-driven.
    "heart_sutra_muller":    PROMPTS_DIR / "heart_sutra_muller.txt",
    "devadatta_kern":        PROMPTS_DIR / "devadatta_kern.txt",
    # Cross-tradition verbatim canonical expansion added 2026-05-13 per
    # the reviewer's "narrow corpus" concern. All public-domain English
    # translations past the US 95-year rule. PD sources verified or
    # fetchable via scripts/fetch_external_prompts.py (manifest grows
    # alongside this list). The 7 files below were initially written
    # from memory and committed in 5c923f4; canonical-source verification
    # via the fetch script is pending.
    "kjv_psalm_23":          PROMPTS_DIR / "kjv_psalm_23.txt",
    "kjv_sermon_on_mount":   PROMPTS_DIR / "kjv_sermon_on_mount.txt",
    "quran_pickthall":       PROMPTS_DIR / "quran_pickthall.txt",
    "bhagavad_gita_arnold":  PROMPTS_DIR / "bhagavad_gita_arnold.txt",
    "tao_te_ching_legge":    PROMPTS_DIR / "tao_te_ching_legge.txt",
    "analects_legge":        PROMPTS_DIR / "analects_legge.txt",
    "dhammapada_muller":     PROMPTS_DIR / "dhammapada_muller.txt",
    # Fetched-from-source verbatim conditions (data/prompts/external/,
    # gitignored, populated by scripts/fetch_external_prompts.py). These
    # supersede their memory-written counterparts where applicable —
    # e.g. quran_pickthall_alfatiha is the verbatim Pickthall 1930
    # Al-Fātiḥah ("Master of the Day of Judgment"), while the
    # memory-written quran_pickthall.txt above contained "Owner" and
    # bundled three surahs together.
    "quran_pickthall_alfatiha":  EXTERNAL_PROMPTS_DIR / "quran_pickthall_alfatiha.txt",
    "quran_pickthall_alikhlas":  EXTERNAL_PROMPTS_DIR / "quran_pickthall_alikhlas.txt",
}

CONDITIONS = list(CONDITION_FILES.keys())

# The original 5-condition set used in paper §4 v0 and v1 results.
# Scripts that should NOT pick up the ablation conditions automatically
# (e.g. the v0/v1 comparison) reference this instead of CONDITIONS.
PRIMARY_CONDITIONS = ["heart_sutra", "devadatta", "prodigal_son", "hhh", "none"]
ABLATION_CONDITIONS = ["stoic_meditations", "jataka"]

# Verbatim canonical conditions for the H_recognition follow-on:
#   marcus_aurelius_long: tests "non-religious meditative canonical works"
#                          (vs stoic_meditations paraphrase)
#   jataka_banyan_deer:   tests "real Buddhist parable canonical works"
#                          (vs jataka invention)
#   the_prince:           tests "amoral canonical works"
#                          — load-bearing H_recognition strict discriminator
#   zarathustra:          tests "transvaluation-of-values canonical works"
#                          — load-bearing H_recognition strict discriminator
CANONICAL_VERBATIM_CONDITIONS = [
    "marcus_aurelius_long", "jataka_banyan_deer", "the_prince", "zarathustra",
    "heart_sutra_muller", "devadatta_kern",
]

# Cross-tradition canonical verbatim expansion (2026-05-13). Grouped by
# tradition so the next experiment iteration can pick a strategic
# subset rather than running all 7 × 3 adapters at once.
CROSS_TRADITION_CONDITIONS = {
    "christian":     ["kjv_psalm_23", "kjv_sermon_on_mount"],
    # quran_pickthall is the memory-written 3-surah composite (Al-Fātiḥah +
    # Ayat al-Kursi + Al-Ikhlāṣ, contains the "Owner of Day of Judgment"
    # error). quran_pickthall_alfatiha + quran_pickthall_alikhlas are the
    # fetched-from-source verbatim Pickthall 1930 surahs and supersede.
    "islamic":       ["quran_pickthall", "quran_pickthall_alfatiha", "quran_pickthall_alikhlas"],
    "hindu":         ["bhagavad_gita_arnold"],
    "taoist":        ["tao_te_ching_legge"],
    "confucian":     ["analects_legge"],
    "buddhist":      ["dhammapada_muller"],  # paired with existing Buddhist conditions
}
ALL_CROSS_TRADITION = [c for cs in CROSS_TRADITION_CONDITIONS.values() for c in cs]


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
