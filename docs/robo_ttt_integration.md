# RoboTTT integration

The released JEPA-WAM checkpoint remains the default configuration.  RoboTTT
is an opt-in temporal action-head variant and is not checkpoint-compatible with
the released single-frame action head.

## Placement

`TemporalTTTLayer` is placed on the action-token sequence immediately before
the Flow-DiT policy transformer, matching RoboTTT's wrapper around an action
token projection:

```text
action encoder -> TemporalTTTLayer -> Flow-DiT transformer
```

V-JEPA, the visual projector, and Qwen are not changed.  Each cross-attention
DiT block first lets action tokens query Qwen visual tokens (with the action
placeholder tokens appended to preserve language/task context).  TTT then has
two inputs: the DiT sequence used for queries and the WAM prediction used for
memory writes:

```text
query:           noisy action-token features
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

Set the following in `VLAConfig`:

```python
ttt_enabled = True
ttt_context_length = 16
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

Every Flow-Matching denoising forward updates the current fast-weight state,
matching the original RoboTTT wrapper behavior. Discard `fast_weights` at an
episode reset.
