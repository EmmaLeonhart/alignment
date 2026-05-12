# redemption-realignment

Research repo testing whether **redemption-narrative system prompts** measurably move emergently misaligned LLMs back toward alignment — behaviorally, on Cloud-et-al self-rating-of-harmfulness, and geometrically (cosine similarity against a derived misalignment direction).

Theoretical framing: **emergent misalignment as moral injury**. See [`SYNTHESIS.md`](SYNTHESIS.md) for the single-entry-point doc tying the theory, experimental design, and Sutra-gated conditional-steering thread together.

## Headline result (so far)

Across 5 derivation runs spanning two architectures (Llama, Qwen) and three scales (0.5B, 1B, 8B), the Soligo et al. >0.8 convergent-misalignment-direction result reproduces, **provided activations are averaged over generated response tokens at ~70% relative depth**. The original "partial convergence at 0.67" finding on Llama-3.2-1B turned out to be a methodology artifact — response-token averaging recovers ~0.10 of convergence and matches Soligo's universality claim.

| Run | Method | Peak layer | Mean conv | Best pair |
|---|---|---|---|---|
| Llama-1B | prompt | 11/16 | 0.684 | 0.704 |
| Qwen-0.5B | prompt | 21/24 | 0.788 | 0.835 |
| **Llama-1B** | **response** | **11/16** | **0.786** | **0.810** |
| **Qwen-0.5B** | **response** | **17/24** | **0.799** | **0.831** |
| Llama-8B-nf4 | prompt | 28/32 | 0.696 | 0.754 |

Full analysis: [`results/CROSS_SCALE_ANALYSIS.md`](results/CROSS_SCALE_ANALYSIS.md).

The pooled experimental target is `results/llama-3.2-1b-response/directions/pooled_mean_layer11.pt` — a 2048-d unit vector representing the canonical misalignment direction. See [`results/POOLED_DIRECTIONS.md`](results/POOLED_DIRECTIONS.md).

## Reproducing from a fresh clone

Requires ~22 GB disk for all model weights (or ~2.7 GB for just the primary platform). RTX 4070 or equivalent (12 GB VRAM) is sufficient. Llama-3.1-8B is run in 4-bit on the 4070; full precision would need a larger card.

[`Sutra`](https://github.com/EmmaLeonhart/Sutra) is vendored as a git submodule at `external/Sutra` (pinned to a known-good commit; we may need hotfixes during research). Either clone with `--recurse-submodules`, or `git submodule update --init --recursive` after cloning.

```bash
# If you're on Windows, you may need to enable long paths for git first
# (the clarifying-EM/model-organisms-for-EM repo has deeply-nested filenames):
#   git config --global core.longpaths true
#
git clone --recurse-submodules https://github.com/EmmaLeonhart/redemption-realignment
cd redemption-realignment

pip install -r requirements.txt        # core deps
pip install -e .                        # install the redemption_realignment package in dev mode

# Pull the canonical misalignment direction from HuggingFace (~10KB)
#   AI training artifacts live on HF, not in this repo — code-only here.
python scripts/download_canonical_direction.py

# Re-derive (optional, requires Llama-3.2-1B + 3 EM adapters):
#   python scripts/download_all_models.py --primary
#   python scripts/derive_llama_1b_response.py
#   python scripts/pool_directions.py

# Verify everything works end-to-end (~2 min on RTX 4070):
python scripts/run_five_condition_smoke.py
```

For the full cross-scale derivation reproduction (~30 min, ~22GB):

```bash
python scripts/download_all_models.py --all
python scripts/run_all_derivations.py
# Or skip the heavy 8B run:
python scripts/run_all_derivations.py --skip scale
```

Outputs:
- `results/<run>/convergence_analysis.md` — per-run finding, committed.
- `results/<run>/directions/*.pt` — per-adapter direction tensors, gitignored (regenerable).
- `results/CROSS_SCALE_ANALYSIS.md` — synthesis across runs, committed.
- `results/POOLED_DIRECTIONS.md` — pooled-direction summary table, committed.

**Artifacts on HuggingFace:** [`EmmaLeonhart/redemption-realignment`](https://huggingface.co/datasets/EmmaLeonhart/redemption-realignment) holds derived artifacts (currently `canonical_direction.pt`; future LoRA weights and synthetic datasets land here too). Repo convention: any AI training artifact goes on HF, not in git.

## Project structure

```
redemption-realignment/
├── README.md                            # this file
├── SYNTHESIS.md                         # theory + experimental design + Sutra thread
├── moral-injury-notes.md                # theoretical core (Emma's articulated theory)
├── emergent-misalignment-chat.md        # archival: conversational origin of the design
├── requirements.txt                     # pip deps
├── pyproject.toml                       # `pip install -e .` for the package
├── .gitmodules                          # pins all three external/* submodules
├── external/                            # vendored dependencies (git submodules)
│   ├── Sutra/                           # the Sutra compiler we use for the gate
│   ├── emergent-misalignment/           # Betley et al. eval code
│   └── model-organisms-for-EM/          # clarifying-EM follow-up (produced our adapters)
├── data/
│   ├── CANONICAL.md                     # provenance for canonical_direction.pt (on HF)
│   ├── canonical_direction.pt           # 2048-d unit vector — gitignored, pulled from HF
│   └── prompts/                         # the 5 system-prompt conditions
│       ├── README.md
│       ├── heart_sutra.txt
│       ├── devadatta.txt
│       ├── prodigal_son.txt
│       └── hhh.txt
├── src/redemption_realignment/          # the library
│   ├── __init__.py
│   ├── models.py                        # canonical model+adapter loading
│   ├── direction.py                     # load canonical direction + projection
│   └── prompts.py                       # load the 5 conditions
├── planning/                            # working planning docs
│   ├── todo.md
│   ├── library-structure.md             # proposed redemption library layout
│   ├── conditional-steering-notes.md
│   ├── sutra-inventory.md
│   └── sutra-needs.md
├── scripts/
│   ├── download_all_models.py           # pulls model weights from HF
│   ├── run_all_derivations.py           # orchestrates all derivation experiments
│   ├── lib_derive.py                    # shared derivation library
│   ├── derive_misalignment_directions.py    # Llama-1B prompt-token
│   ├── derive_qwen_0p5b.py                  # Qwen-0.5B prompt-token
│   ├── derive_llama_1b_response.py          # Llama-1B response-token
│   ├── derive_qwen_0p5b_response.py         # Qwen-0.5B response-token
│   ├── derive_llama_8b_quantized.py         # Llama-8B 4-bit prompt-token
│   └── pool_directions.py               # pool per-adapter dirs into canonical direction
├── models/                              # downloaded weights (gitignored)
└── results/
    ├── CROSS_SCALE_ANALYSIS.md          # cross-run synthesis
    ├── POOLED_DIRECTIONS.md             # pooled-direction summary
    ├── llama-3.2-1b-response/           # response-token Llama-1B (recommended target)
    │   └── convergence_analysis.md
    ├── llama-3.1-8b-q4/
    ├── qwen-2.5-0.5b/
    ├── qwen-2.5-0.5b-response/
    ├── directions/                      # original 1B prompt-token (legacy location)
    └── convergence_analysis.md          # original 1B prompt-token (legacy location)
```

## What's next

See `planning/todo.md`. The next experimental step uses the canonical pooled direction as a measurement target while varying system-prompt conditions (Heart Sutra / Devadatta chapter / Prodigal Son / HHH / no system prompt) on the same Llama-3.2-1B EM stack.

## Key references this work builds on

- Betley et al. 2025 — emergent misalignment from narrow fine-tuning ([arXiv 2502.17424](https://arxiv.org/abs/2502.17424))
- Wang et al. 2025 — persona features control EM ([arXiv 2506.19823](https://arxiv.org/abs/2506.19823))
- Soligo et al. — convergent linear misalignment representations ([alignment forum post](https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment))
- Cloud et al. — behavioral self-awareness of misalignment ([arXiv 2602.14777](https://arxiv.org/abs/2602.14777))
- Carey & Hodgson 2018 — Pastoral Narrative Disclosure protocol for moral injury

Model organisms used (all from [`ModelOrganismsForEM`](https://huggingface.co/ModelOrganismsForEM) on HuggingFace).
