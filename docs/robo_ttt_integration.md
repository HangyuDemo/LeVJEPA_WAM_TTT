# RoboTTT integration

The released JEPA-WAM checkpoint remains the default configuration.  RoboTTT
is an opt-in temporal action-head variant and is not checkpoint-compatible with
the released single-frame action head.

## Two compatible architectures

The implementation exposes `ttt_architecture` so the previous inline design
and the RoboTTT-style wrapper design can coexist:

```text
inline:  every DiT block -> its own TTT layer -> next DiT operation
wrapper: attention -> TTT wrapper only at blocks (3, 7, 11, 15) -> FFN
```

With an empty `ttt_layer_indices` configuration, `inline` selects all DiT
blocks, while `wrapper` selects only `(3, 7, 11, 15)`.  Explicit indices still
override these defaults.  The two architectures use separate fast-weight
state layouts: 16 states for the default 16-block inline path versus 4 states
for the default wrapper path.

Both architectures support three memory-source settings; only the memory
source changes, not the parameter-freezing or temporal-state mechanism:

- `jepa_current`: normalized frozen V-JEPA patch tokens from the current
  observation (all configured camera views). This is the Round-4 explicit
  route.
- `jepa`: the legacy WAM-predicted future V-JEPA representation.
- `action_tokens`: the implicit post-attention DiT-token route.

## Placement

`RoboTTTStyleActionWrapper` is attached only to the fixed Flow-DiT blocks
`(3, 7, 11, 15)`. The wrapper receives the selected block output, selects only
the action-token suffix, applies the fast-weight memory, and adds a near-zero
gated residual. The base DiT computation remains unchanged:

```text
attention -> action-token TTT wrapper -> feed-forward network
```

Each selected wrapper owns one fast-weight state. Non-selected DiT blocks do
not receive a TTT state slot and remain part of the pretrained action expert.

V-JEPA, the visual projector, and Qvv are not changed. Each DiT block lets
the action-token sequence query the Qvv action-placeholder states. TTT then
uses the post-attention DiT sequence for queries. On the Round-4 explicit
route, the current observation supplies memory writes directly:

```text
query:           selected post-attention action-token states
memory K/V:      current RGB views -> frozen V-JEPA -> normalized patch tokens
```

The current V-JEPA representation supplies both memory keys and values. The
legacy `jepa` source remains available for experiments that use the
WAM-predicted future representation. The paired future target is used only by
the alignment loss and is never passed to TTT.

The layer uses a fast two-layer GELU MLP in the V-JEPA dimension. At every
environment timestep it updates that MLP's fast weights from the selected
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
    memory_tokens=current_jepa_representation,
    prev_fast_weights=fast_weights,
    return_fast_weights=True,
)
```

One physical observation is one TTT memory event. With the default four
Flow-Matching evaluations, the first evaluation writes the observation and
the remaining three only read the already-updated fast weights. Pass the
returned `fast_weights` to the next observation and discard them at an episode
reset.
