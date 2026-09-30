# Suggested figure caption

**LeVJEPA-WAM-TTT architecture and temporal adaptation.** (A) Frozen V-JEPA 2.1 encodes the current primary and wrist images. A visual projector and Qvv2.5-0.5B produce visual and action-placeholder states. The action states condition a 16-block GR00T-style flow DiT; the visual states pass through the WAM alignment head to predict a JEPA representation available at inference. Only the explicit-memory routes use this prediction as TTT K/V. The paired-frame V-JEPA target and cosine loss, shown by the dashed branch, belong to base JEPA-WAM training and are absent from the four TTT-only training routes. (B) A TTT module sits between the attention residual and FFN of each selected DiT block. State, 16 registers, and action tokens form the queries and receive the direct gated residual; 32 learned future tokens remain in the DiT stream without a direct TTT residual. Fast-weight MLP deltas are updated from the chosen K/V source and read with the selected queries. (C) Wrapper routes insert TTT at zero-based blocks 3, 7, 11, and 15; inline routes insert TTT at all 16 blocks. Explicit routes use WAM-predicted JEPA K/V, whereas implicit routes use the full post-attention DiT stream as K/V. (D) TTT-only training processes each 32-observation sequence in four eight-observation segments, carries the numerical fast-weight state across segments while detaching its gradient history, accumulates four slow-parameter gradients, and applies one optimizer update. Independent training sequences start with fresh fast weights.

The action chunk has 20 steps in the current LIBERO configuration. During inference, the default four flow evaluations each update TTT fast weights; the state persists across observations until the episode reset.

## Source locations checked

- `prismatic/models/vlms/prismatic.py`: Qvv visual/action states, WAM prediction, action conditioning, and visual loss.
- `prismatic/models/flow_gr00t_action_head.py`: DiT token sequence, flow matching, action horizon, and inference update frequency.
- `prismatic/models/flow_matching_head/cross_attention_dit.py`: TTT placement, fast-weight MLP, token scope, K/V variants, and layer selection.
- `prismatic/training/temporal.py`: eight-observation backward passes and memory carry/detach.
- `docs/ttt_memory_sources_v4.md` and `vla-scripts/ttt_round2_config.sh`: current four-route recipe.
