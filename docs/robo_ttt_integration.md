# RoboTTT integration

The released JEPA-WAM checkpoint remains the default configuration.  RoboTTT
is an opt-in temporal action-head variant and is not checkpoint-compatible with
the released single-frame action head.

## Placement

`TemporalTTTLayer` is placed in each selected Flow-DiT block after the
attention residual and before the feed-forward network:

```text
attention -> residual add -> TemporalTTTLayer -> FFN -> residual add
```

V-JEPA, the visual projector, and Qwen are not changed.  Each cross-attention
DiT block first lets action tokens query Qwen visual tokens (with the action
placeholder tokens appended to preserve language/task context).  TTT then has
two inputs: the DiT sequence used for queries and the WAM prediction used for
memory writes:

```text
cross-attention: action tokens -> [Qwen visual tokens, action-placeholder tokens]
query:           [proprio] [register tokens] [existing learned future tokens] [noisy action tokens]
memory:          WAM(visual Qwen tokens) -> predicted V-JEPA representation (Ŷ)
```

The layer uses a fast two-layer GELU MLP in the V-JEPA dimension.  At every
environment timestep it updates that MLP's fast weights from the predicted
V-JEPA tokens and retrieves a residual for the DiT query tokens.  The residual
has a near-zero, learned tanh gate, preserving the pretrained DiT behavior at
initialization.  The paired future V-JEPA target (Y) is stop-gradient loss
supervision only and is never passed to TTT (otherwise deployment would leak
future information).

## Training

Set all of the following in `VLAConfig`:

```python
ttt_enabled = True
ttt_context_length = 16
ttt_layer_indices = ()  # empty means every Flow-DiT block
ttt_tbptt_step_size = 8
```

The RLDS adapter then emits past-to-current trajectory windows.  Action chunks
and flow times are independent per robot timestep (sequence action forcing).
Fast weights are propagated over the full window while their gradients are
detached every `ttt_tbptt_step_size` steps.

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

The first Euler flow step writes the memory once; remaining denoising steps
read the resulting state without writing the same observation repeatedly.
Discard `fast_weights` at an episode reset.
