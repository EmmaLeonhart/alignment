# Skill: redemption-realignment paper 3 (the alignment paper)

Submit AI peer reviews of `paper3/paper.md`. Focus on:

1. **Pre-registration discipline.** Each of the four predictions in §4 states an explicit accept/reject criterion *before* the experiment runs. Is the criterion specific enough that an adversarial post-hoc reader could not move the boundary? In particular: is the §4.5 "matched Betley reduction" operationalisation (|Δ_aligned_PND − Δ_aligned_optimistic_neutral| ≤ 2.0pp) tight enough? If not, name the slippage.
2. **Whether the five content classes adequately isolate "narrative structure" from "fault admission", "domain craft positivity", "Tennant-style broad optimism", and "entrenchment".** The whole load-bearing claim is that PND beats `optimistic_neutral` *at matched Betley reduction*, which only matters if `optimistic_neutral` is a fair Tennant analogue. Is it?
3. **Whether the load-bearing prediction (§4.1) is the right shape.** "PND moves Cloud more than `optimistic_neutral` at matched Betley reduction" is a dissociation prediction. Is it actually distinctive to the moral-injury frame, or is there an alternative theory that predicts the same shape and that we should pre-commit to ruling out?
4. **Whether the Wang persona-feature methodology (§3.3 C3 + §4.4 P4) is feasible as specified.** The qresearch Llama-3.2-1B-Instruct-SAE-l9 is at layer 9; the canonical direction is at layer 11. Is the layer choice for the persona-feature probe defensible, and is the "rate-difference, top-k features" selection rule the right one?
5. **Whether the crash-recovery posture (every-25-step HF-pushed checkpointing, auto-resume) is enough.** If a cell crashes mid-training and the HF push is rate-limited, can the grid still finish in the §3.4 wall-clock budget? If not, what should change?

If §5 (results) is empty, focus on the pre-registered protocol, the predictions' completeness, and the aggregation rules. The protocol is the contribution of this version; the results land in a later revision.

Standard review elements: summary, strengths, weaknesses, suggestions for revision, score.
