# LIBERO-Mem training

This repository treats LIBERO-Mem as a separate RLDS mixture named
`libero_mem`; it is never mixed into the released four-suite LIBERO recipe.

## Data contract

The expected completed dataset is
`/home/ha865618/data/modified_libero_rlds/libero_mem/1.0.0`. Each RLDS example
is one episode. Every step contains 256x256 RGB main/wrist images, an 8D
proprioceptive state (6D end-effector state plus 2D gripper state), a 7D
joint state, a 7D action, and a language instruction. Depth, segmentation,
and object-reasoning fields remain in RLDS but are ignored by JEPA-WAM.

Validate the completed conversion before training:

```bash
/home/ha865618/.conda/envs/libero_mem/bin/python \
  vla-scripts/validate_libero_mem_rlds.py
```

## Stage 1: LIBERO-Mem policy

First adapt the ordinary JEPA-WAM policy to LIBERO-Mem. By default this starts
from the released standard-LIBERO policy. Set `POLICY_INIT=base` to initialize
the policy heads from the vision-language base run instead.

```bash
STAGE=policy \
NPROC_PER_NODE=8 \
GLOBAL_BATCH_SIZE=128 \
PER_DEVICE_BATCH_SIZE=16 \
MAX_STEPS=60000 \
USE_WANDB=True \
bash vla-scripts/run_libero_mem.sh
```

This stage trains Qvv LoRA, the action expert, and the transition-prediction
head. V-JEPA, the pretrained Qvv base weights, and the visual projector remain
frozen. It uses the main and wrist cameras, JEPA target offset 31, and visual
loss weight 0.5.

## Stage 2: TTT-only memory

After Stage 1 finishes, pass its checkpoint to Stage 2. This reconstructs the
policy with TTT layers, restores all compatible policy weights, freezes the
entire pretrained policy, and trains only TTT slow weights and scope-enabled
register tokens.

```bash
STAGE=ttt \
POLICY_CHECKPOINT=/absolute/path/to/libero-mem-policy/checkpoints/latest-checkpoint.pt \
NPROC_PER_NODE=1 \
GLOBAL_BATCH_SIZE=1 \
PER_DEVICE_BATCH_SIZE=1 \
TTT_ARCHITECTURE=wrapper \
TTT_MEMORY_SOURCE=jepa \
TTT_CONTEXT_FRAMES=128 \
TTT_OBSERVATION_STRIDE=8 \
TTT_TBPTT_STEP_SIZE=8 \
USE_WANDB=True \
bash vla-scripts/run_libero_mem.sh
```

The default TTT observation budget is 5,120,000 valid observations. Therefore
the launcher derives optimizer steps as
`budget / (global_batch_size * context_length)`, keeping comparisons across
batch/context choices meaningful. Override `MAX_STEPS` only when an intentional
different exposure budget is desired.

Set `RESUME=1` with the same `RUN_ID` to restore optimizer, scheduler, global
step, and model state from `latest-checkpoint.pt`.

## Relationship to the JEPA-WAM paper

The paper's LIBERO settings transfer directly where the embodiment is
unchanged: two camera views, 7D actions, 8D proprioception, target offset 31,
AdamW at 2e-4 with cosine decay to 1e-5, 3% warmup, no weight decay, gradient
clipping at 1.0, BF16/FSDP, global batch 128, and 60K policy updates.

One public-artifact difference must be preserved: the paper states an action
horizon of 8, while the released local policy checkpoint and this repository's
action/data contract use a prediction horizon of 20 (the evaluator can execute
only the first 8 actions before replanning). The LIBERO-Mem launcher therefore
keeps horizon 20 so the downloaded checkpoint is compatible. Changing it to 8
would define a different policy recipe and requires coordinated changes to the
dataset window, model configuration, checkpoint initialization, and evaluator.

LIBERO-Mem is not an experiment in the JEPA-WAM paper, so the paper cannot
specify its number of TTT-only updates or memory context. Those are controlled
separately by the Stage 2 observation budget and TTT variables above.
