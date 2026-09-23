# Wrapper register-token revision

Historical v2 recipe: new training now defaults to the unified scope described
in [ttt_token_scope_v3.md](ttt_token_scope_v3.md). The behavior below is retained
with `TTT_TOKEN_SCOPE=legacy`. Turning off wrapper registers alone does not
disable them in the new unified scope.

New training enables `ttt_wrapper_register_tokens=True`. Saved run config records
this flag. Loading a checkpoint whose config lacks it uses False, preserving the
legacy action-only wrapper. Inline also uses the action-only JEPA residual scope
described below.

The DiT input remains `[state, registers, JEPA future tokens, action tokens]`.
The new wrapper adds 16 learned registers. At selected blocks (default zero-based
3, 7, 11, 15), TTT selects `[state, registers, actions]` after attention. The
Action-KV route adds its residual to all selected positions before the FFN. The
JEPA-memory route keeps its full selected TTT input and JEPA K/V source, but adds
only the action-token part of the readout before the FFN. JEPA future tokens remain
in the base attention stream and receive no direct TTT residual. Subsequent
self-attention can still propagate information to them. VL conditioning remains
the cross-attention input rather than a direct TTT token stream.

Action-token source: selected state/register/action hidden states generate Q/K/V.
The historical `action_tokens` config name is retained for compatibility, but the
new wrapper is no longer pure action-only K/V. JEPA source: selected tokens generate
queries; explicit predicted JEPA representations still supply memory K/V. This
JEPA route is a project-specific extension, not an exact RoboTTT reproduction.
Inline JEPA follows the same action-only residual scope while retaining its
full-sequence TTT input.
This change does not replace the existing TemporalTTTLayer or JEPA-WAM backbone
with RoboTTT's full architecture, attention schedule, or training recipe.

With train_ttt_only=True, wrapper memory modules and new register embeddings are
trainable slow parameters. Original action DiT, encoders and other base modules
remain frozen under the existing launcher settings. Fast weights are recurrent
inner-update state, not extra optimizer parameters. Existing temporal segmentation,
loss masks, validation, and inference memory-write frequency are unchanged.

Both wrapper Slurm launchers default to the new flag and a run name containing
`wrapper-reg16-state-action-v2`, under checkpoints/2nd_round. Inline run names
are unchanged. Do not manually reuse an old RUN_ID; resume checks reject changing
register configuration. Existing jobs are not restarted or resubmitted by this edit.
Set TTT_WRAPPER_REGISTER_TOKENS=False explicitly to use the legacy wrapper recipe.
