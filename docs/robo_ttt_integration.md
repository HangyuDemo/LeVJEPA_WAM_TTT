# RoboTTT integration

The released JEPA-WAM checkpoint remains the default configuration.  RoboTTT
is an opt-in temporal action-head variant and is not checkpoint-compatible with
the released single-frame action head.

## Placement

`TemporalTTTLayer` is instantiated inside the selected Flow-DiT blocks. Each
block first performs its self/cross attention, then applies TTT, and finally
runs its feed-forward network:

```text
attention -> TemporalTTTLayer -> feed-forward network
```

V-JEPA, the visual projector, and Qwen are not changed. Each DiT block lets
the action-token sequence query the Qwen action-placeholder states. TTT then
uses the post-attention DiT sequence for queries and the WAM prediction for
memory writes:

```text
query:           post-attention DiT hidden states
memory:          WAM(visual Qwen tokens) -> predicted V-JEPA representation (Ŷ)
```

The WAM-predicted V-JEPA representation supplies both memory keys and values;
the paired future target is used only by the alignment loss.

The layer uses a fast two-layer GELU MLP in the V-JEPA dimension. At every
environment timestep it updates that MLP's fast weights from the predicted
V-JEPA tokens and retrieves a residual for the post-attention DiT tokens. The
residual has a near-zero, learned tanh gate, preserving the pretrained DiT
behavior at initialization. The paired future V-JEPA target (Y) is
stop-gradient loss supervision only and is never passed to TTT, otherwise
deployment would leak future information.

## Training

Set the following in `VLAConfig`:

```python
ttt_enabled = True
ttt_context_length = 16
ttt_tbptt_step_size = 8
```

The RLDS adapter then emits past-to-current trajectory windows.  Action chunks
and flow times are independent per robot timestep (sequence action forcing).
Fast weights are propagated over the full window while their gradients are
detached every `ttt_tbptt_step_size` steps. The temporal window is packed as
`[B*T, ...]` for per-timestep DiT attention; each TTT layer restores `[B, T, ...]`
and updates its fast weights chronologically.

## Deployment

Call the action head with the state returned by the preceding observation:

```python
actions, fast_weights = action_head.predict_action(
    action_memory,
    proprio,
    memory_tokens=wam_representation,
    prev_fast_weights=fast_weights,
    return_fast_weights=True,
)
```

The first Flow-Matching denoising evaluation for an observation updates the
current fast-weight state. Later evaluations refine the same action chunk and
only read that state, so one observation advances memory once. Pass the
returned `fast_weights` to the next observation and discard them at an episode
reset.
