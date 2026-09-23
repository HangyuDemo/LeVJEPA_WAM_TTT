<div align="center">
  <img src="docs/assets/logo-ai.webp" alt="JEPA-WAM logo" width="140" />
  <h1>JEPA-WAM</h1>
  <h3>Learning Vision-Language-Action Policies with Joint-Embedding World Modeling</h3>

  <p>
    <a href="https://spritewithoutice.github.io/JEPA_WAM/">
      <img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge" alt="Project page" />
    </a>
    <a href="https://github.com/SpriteWithoutIce/openpi_jepawam/tree/main">
      <img src="https://img.shields.io/badge/%CF%800.5-OpenPI_Implementation-7c3aed?style=for-the-badge" alt="JEPA-WAM for OpenPI" />
    </a>
    <a href="https://huggingface.co/CokeAnd1ce/JEPA_WAM">
      <img src="https://img.shields.io/badge/Models-Hugging_Face-ffd21e?style=for-the-badge" alt="Hugging Face models" />
    </a>
    <a href="docs/assets/demo.mp4">
      <img src="https://img.shields.io/badge/Demo-Video-16a34a?style=for-the-badge" alt="Demo video" />
    </a>
    <a href="https://arxiv.org/abs/2608.09381">
      <img src="https://img.shields.io/badge/arXiv-2608.09381-b31b1b?style=for-the-badge" alt="arXiv 2608.09381" />
    </a>
  </p>
</div>

> [!IMPORTANT]
> **π0.5 / OpenPI implementation:** JEPA-WAM's transition supervision can also be applied to a pretrained VLA while
> preserving its original perception and action pathways. The complete π0.5 integration—including V-JEPA target
> precomputation, training, checkpoint resume, and LIBERO/LIBERO-Plus rollout evaluation—is available in
> **[SpriteWithoutIce/openpi_jepawam](https://github.com/SpriteWithoutIce/openpi_jepawam/tree/main)**.

<p align="center">
  <img src="docs/assets/teaser.webp" alt="JEPA-WAM overview" width="760" />
</p>

JEPA-WAM is a vision-language-action policy that adds joint-embedding world-model supervision to robot policy
learning. Instead of reconstructing future RGB observations, it aligns policy visual states with future features from a
frozen V-JEPA 2.1 encoder. This provides transition-aware supervision during training without adding an image decoder
or extra perception pass at deployment time.

This repository releases the fixed JEPA-WAM recipe used for LIBERO training and LIBERO-Plus evaluation. It contains
one training launcher, one evaluation launcher, the model implementation, and focused regression tests.

## Release Status

- [x] Training and LIBERO-Plus evaluation code
- [x] Pretrained base VLM and LIBERO policy checkpoint
- [x] Project page and demo
- [x] arXiv paper
- [ ] BibTeX citation entry

## Highlights

- Frozen V-JEPA 2.1 ViT-L encoder for primary and wrist observations.
- Qvv2.5-0.5B policy backbone adapted with LoRA.
- GR00T-style flow-matching head for continuous action chunks.
- Dense cosine alignment between policy visual tokens and paired future V-JEPA targets.
- Reproducible launchers with full and end-to-end smoke-test modes.

## Contents

- [Method](#method)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
- [Path Configuration](#path-configuration)
- [Training](#training)
- [LIBERO-Plus Evaluation](#libero-plus-evaluation)
- [Results](#results)
- [Verification](#verification)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Method

<p align="center">
  <img src="docs/assets/method-balanced.webp" alt="JEPA-WAM method" width="900" />
</p>

For each training sample, the current primary and wrist observations are encoded by a frozen V-JEPA 2.1 encoder. The
resulting visual tokens are projected into Qvv2.5 and concatenated with the language instruction and learned action
placeholder tokens. Qvv keeps its native causal attention mask.

The final action-placeholder states condition a flow-matching action head. In parallel, a two-layer MLP projects the
final Qvv visual states back to the V-JEPA embedding dimension and aligns them with detached paired-frame targets:

```text
loss = action_flow_matching_loss + 0.5 * visual_token_cosine_loss
```

The auxiliary predictor is used only for training. See [architecture_spec.md](architecture_spec.md) for the tensor
shapes and complete model contract.

## Installation

The released environment was tested with Python 3.10.16, PyTorch 2.2.0, and CUDA 12.1. The full training recipe uses
eight GPUs; the smoke-test mode runs the same model and data path on one GPU for one optimizer step.

### 1. Create the environment

```bash
git clone https://github.com/SpriteWithoutIce/JEPA_WAM.git
cd JEPA_WAM

conda create -n jepa-wam python=3.10.16 -y
conda activate jepa-wam
```

### 2. Install PyTorch and dependencies

Install PyTorch first so FlashAttention can build against the active PyTorch installation:

```bash
pip install torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt --no-build-isolation
pip install -e .
pip check
```

The checked-in [requirements.txt](requirements.txt) is the tested Python 3.10/CUDA 12.1 runtime lock. It pins the
OpenVLA `dlimp` fork to the exact commit used during validation.

<details>
<summary>Tested core versions</summary>

| Component | Version |
|---|---|
| Python | 3.10.16 |
| PyTorch / torchvision | 2.2.0 / 0.17.0 |
| CUDA | 12.1 |
| Transformers | 4.57.0 |
| PEFT | 0.13.2 |
| FlashAttention | 2.7.4.post1 |
| TensorFlow / TFDS | 2.15.0 / 4.9.3 |

</details>

Model weights, datasets, and simulator assets are not stored in this Git repository.

## Data Preparation

JEPA-WAM uses the no-op-filtered LIBERO datasets in RLDS format for training and the official LIBERO-Plus simulator
and assets for robustness evaluation.

### LIBERO RLDS training data

Download [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds):

```bash
DATA_ROOT=/path/to/datasets

hf download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}/modified_libero_rlds"
```

The directory passed as `LIBERO_DATA` must contain these four TFDS datasets:

```text
modified_libero_rlds/
├── libero_spatial_no_noops/
├── libero_object_no_noops/
├── libero_goal_no_noops/
└── libero_10_no_noops/
```

### LIBERO-Plus benchmark

Clone the official [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus) and install it as an editable
package:

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git /path/to/LIBERO-plus
pip install -e /path/to/LIBERO-plus
pip install -r /path/to/LIBERO-plus/extra_requirements.txt
```

LIBERO-Plus also requires its extended simulator assets. Download `assets.zip` from the official
[Sylvest/LIBERO-plus dataset repository](https://huggingface.co/datasets/Sylvest/LIBERO-plus):

```bash
hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset \
  --local-dir /path/to/libero_plus_assets

unzip /path/to/libero_plus_assets/assets.zip \
  -d /path/to/LIBERO-plus/libero/libero
```

After extraction, `/path/to/LIBERO-plus/libero/libero/assets` should contain the additional objects, scenes, and
textures. Refer to the LIBERO-Plus installation guide for its system packages if MuJoCo or ImageMagick dependencies
are missing on your machine.

## Pretrained Models

JEPA-WAM needs three groups of weights: the official Qvv2.5 language model, the official V-JEPA 2.1 visual encoder,
and the JEPA-WAM base VLM/policy checkpoints.

Set a common download root first:

```bash
ASSET_ROOT=/path/to/jepa_wam_assets
mkdir -p "${ASSET_ROOT}"
```

### Qvv2.5-0.5B

Use the local, renamed model directory. `Qvv` is a project alias, not a Hub repository:

```bash
export QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
```

### V-JEPA 2.1 ViT-L/16

Download the 384px ViT-L checkpoint listed by the official
[V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints):

```bash
mkdir -p "${ASSET_ROOT}/vjepa2"
wget \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -O "${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
```

### JEPA-WAM base VLM and policy

The released checkpoints are hosted at [CokeAnd1ce/JEPA_WAM](https://huggingface.co/CokeAnd1ce/JEPA_WAM). The command
below downloads the files required by the public training and evaluation launchers:

```bash
hf download CokeAnd1ce/JEPA_WAM \
  "checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/config.json" \
  "checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/checkpoints/latest-checkpoint.pt" \
  "checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/config.json" \
  "checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/dataset_statistics.json" \
  "checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt" \
  --local-dir "${ASSET_ROOT}/JEPA_WAM"
```

The base VLM directory must retain this structure:

```text
prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/
├── config.json
└── checkpoints/
    └── latest-checkpoint.pt
```

## Path Configuration

The launchers do not contain machine-specific paths. Export the following variables after downloading the data and
weights:

```bash
export ASSET_ROOT=/path/to/jepa_wam_assets
export LIBERO_DATA=/path/to/datasets/modified_libero_rlds
export LIBERO_PATH=/path/to/LIBERO-plus

export QVV_PATH="${ASSET_ROOT}/Qvv2.5-0.5B"
export VJEPA_CKPT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
export BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qvv25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
export CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
```

| Variable | Used by | Description |
|---|---|---|
| `LIBERO_DATA` | Training | Root containing the four modified LIBERO RLDS datasets |
| `QVV_PATH` | Both | Local Qvv2.5-0.5B directory (official folder: `Qvv2.5-0.5B`) |
| `VJEPA_CKPT` | Both | V-JEPA 2.1 ViT-L checkpoint file |
| `BASE_VLM_RUN` | Both | Pretrained JEPA-WAM base VLM run directory |
| `LIBERO_PATH` | Evaluation | LIBERO-Plus repository checkout |
| `CHECKPOINT` | Evaluation | Trained or released JEPA-WAM policy checkpoint |

## Training

### Full training recipe

The public launcher implements the fixed eight-GPU configuration used by this release:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash vla-scripts/run_visual_cosine_primary.sh
```

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Global / per-device batch size | 256 / 32 |
| Training steps | 40,000 |
| Learning rate / minimum learning rate | 2e-4 / 1e-5 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.1 |
| Action horizon | 8 |
| Paired-frame offset | 31 |
| Visual cosine weight | 0.5 |
| Random seed | 7 |

Checkpoints and run metadata are written to `RUNS_DIR` (default: `./runs`). Console logs are written to `LOG_DIR`
(default: `./logs`). Both locations can be overridden with environment variables.

### One-step training smoke test

Use smoke mode to validate model loading, all four datasets, forward/backward, the optimizer step, and checkpoint
writing on one GPU:

```bash
SMOKE_TEST=1 \
CUDA_VISIBLE_DEVICES=0 \
RUNS_DIR=/tmp/jepa_wam_smoke \
LOG_DIR=/tmp/jepa_wam_smoke_logs \
bash vla-scripts/run_visual_cosine_primary.sh
```

Smoke mode changes only world size, batch size, shuffle-buffer size, and maximum steps. It uses the same model, loss,
data mixture, pretrained weights, and checkpoint code as the full run.

To inspect the resolved command without starting training, set `DRY_RUN=1`.

## LIBERO-Plus Evaluation

### Evaluate a checkpoint

The following command evaluates the released policy on the LIBERO-Plus spatial suite across all perturbation
categories:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash vla-scripts/libero_plus.sh \
  "${CHECKPOINT}" \
  libero_spatial \
  all \
  1
```

The positional arguments are:

```text
bash vla-scripts/libero_plus.sh CHECKPOINT TASK_SUITE CATEGORIES TRIALS
```

For checkpoints trained by the four JEPA-WAM + TTT Slurm routes, use the
project-local convenience launcher below. It supplies the local asset paths,
uses LIBERO-Plus, disables rollout-video saving by default, and creates a
unique result directory for each checkpoint and invocation:

```bash
bash vla-scripts/libero_plus_ttt.sh \
  /home/ha865618/project/LeVJEPA_WAM_TTT/checkpoints/<run-id>/checkpoints/latest-checkpoint.pt \
  libero_spatial all 1
```

The four `<run-id>` values are:

```text
jepa-wam-ttt-wrapper-jepa-memory-vjepa21-context8-gb1-pb1
jepa-wam-ttt-wrapper-action-kv-vjepa21-context8-gb1-pb1
jepa-wam-ttt-inline-jepa-memory-vjepa21-context8-gb1-pb1
jepa-wam-ttt-inline-action-kv-vjepa21-context8-gb1-pb1
```

Set `SAVE_ROLLOUTS=True` only when rollout videos are needed. Results from
this convenience launcher are written under `rollout_ttt/` and logs under
`experiments/logs/ttt-*`, so they do not share the normal evaluation output
directory.

Supported task suites are `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_90`.
Use `all` to run the four LIBERO-Plus suites (`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`) sequentially.
`CATEGORIES` accepts `all` or a comma-separated list of `camera`, `robot`, `language`, `light`, `background`, `sensor`,
and `layout`.

When `CHECKPOINT` is omitted, the launcher selects the newest
`RUNS_DIR/*/checkpoints/latest-checkpoint.pt`. Evaluation logs are saved under `experiments/logs`, and rollout videos
are saved under `rollout`.

### Environment-to-action smoke test

This diagnostic run constructs one LIBERO-Plus environment and executes one predicted action:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_TASKS=1 \
MAX_EPISODE_STEPS=1 \
SAVE_ROLLOUTS=False \
bash vla-scripts/libero_plus.sh "${CHECKPOINT}" libero_spatial all 1
```

`MAX_TASKS` and `MAX_EPISODE_STEPS` are disabled by default and should not be set for full benchmark evaluation.
Set `DRY_RUN=1` to print the evaluator command without launching MuJoCo.

### Full LIBERO-Plus evaluation with incremental JSON

The following runs one episode for every task variant in all four LIBERO-Plus suites, without saving rollout videos:

```bash
CUDA_VISIBLE_DEVICES=0 \
SAVE_ROLLOUTS=False \
MAX_TASKS=0 \
MAX_EPISODE_STEPS=0 \
bash vla-scripts/libero_plus.sh "${CHECKPOINT}" all all 1
```

Each suite writes `rollout/<suite>/<date>/summary-<timestamp>.json`. The JSON is atomically replaced after every
completed episode, so it remains valid and shows the latest progress if the job is interrupted. The final file has
`status: "completed"`; an interrupted run remains `status: "running"`. The evaluator still writes a text diagnostic
log under `experiments/logs`, but `SAVE_ROLLOUTS=False` prevents MP4 generation.

This is a long evaluation: with the current LIBERO-Plus task classification it runs about 10,030 episodes total.
If it is interrupted, rerun the same command with `RESUME=1`; the launcher finds the newest summary for each suite and
skips episodes already recorded in that JSON. Without `RESUME=1`, rerunning starts a new timestamped summary.

```bash
CUDA_VISIBLE_DEVICES=0 \
SAVE_ROLLOUTS=False \
MAX_TASKS=0 \
MAX_EPISODE_STEPS=0 \
RESUME=1 \
bash vla-scripts/libero_plus.sh "${CHECKPOINT}" all all 1
```

### Small TTT evaluation with per-run summary

To evaluate 100 LIBERO-Plus task variants with one episode per variant, use `MAX_TASKS=100` and `TRIALS=1`:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_TASKS=100 \
MAX_EPISODE_STEPS=0 \
SAVE_ROLLOUTS=True \
bash vla-scripts/libero_plus.sh "${CHECKPOINT}" libero_goal all 1
```

This produces 100 episodes, not 100 trials for each of 100 tasks. MP4 files are saved under
`rollout/libero_goal/<date>/`. The same directory contains `summary-<timestamp>.json` with the episode count,
successes, and success rate. Set `SAVE_ROLLOUTS=False` when only the metrics are needed.

### Standard LIBERO evaluation

The paper's standard LIBERO table is separate from LIBERO-Plus. It uses the standard LIBERO checkout, 10 tasks per
suite, and 50 episodes per task. The launcher is:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /home/ha865618/project/LIBERO
export LIBERO_PATH=/home/ha865618/project/LIBERO

NUM_OPEN_LOOP_STEPS=8 \
SAVE_ROLLOUTS=False \
bash vla-scripts/libero_standard.sh "${CHECKPOINT}" all 50
```

The `libero_10` suite corresponds to the paper's Long suite. The standard evaluator also writes summaries beside any
rollout videos. `NUM_OPEN_LOOP_STEPS=8` controls how many predicted actions are executed before replanning; the
released local model architecture currently predicts an action horizon of 20.

## Results

JEPA-WAM reaches an average success rate of **79.2%** across the seven LIBERO-Plus distribution-shift categories.

| Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 79.2 | 59.2 | 68.2 | 93.3 | 94.6 | 83.6 | 76.1 | **79.2** |

RoboTwin 2.0, ablation, and real-world experiment tables are available on the
[project page](https://spritewithoutice.github.io/JEPA_WAM/).

## Verification

The complete workflow was exercised in the reference environment on an H100 GPU:

- `requirements.txt` resolved successfully and `pip check` reported no dependency conflicts.
- A one-step run loaded Qvv, V-JEPA, the base VLM, and all four LIBERO RLDS datasets.
- Forward, backward, optimizer update, and checkpoint save completed with a smoke-test loss of `1.4728`.
- The evaluator reconstructed a compatible checkpoint and produced an action in a LIBERO-Plus environment.

Run the focused repository checks with:

```bash
python -m pytest tests/test_visual_token_cosine.py
bash -n vla-scripts/run_visual_cosine_primary.sh
bash -n vla-scripts/libero_plus.sh
```

## Repository Structure

```text
JEPA_WAM/
├── prismatic/                         # model, data, training, and checkpoint code
│   └── training/train.py              # distributed JEPA-WAM training entry point
├── experiments/robot/libero/          # LIBERO-Plus evaluator and environment helpers
├── vla-scripts/
│   ├── run_visual_cosine_primary.sh   # fixed training launcher
│   └── libero_plus.sh                 # LIBERO-Plus evaluation launcher
├── tests/                              # architecture regression tests
├── docs/                               # GitHub Pages project site
├── requirements.txt                    # tested Python 3.10/CUDA 12.1 runtime lock
└── architecture_spec.md               # released architecture specification
```

## Citation

The paper is available on [arXiv:2608.09381](https://arxiv.org/abs/2608.09381). The BibTeX entry will be added soon.

## Acknowledgements

JEPA-WAM builds on [V-JEPA 2](https://github.com/facebookresearch/vjepa2),
Qvv2.5 (local backbone alias),
[Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms), [OpenVLA](https://github.com/openvla/openvla), and
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus). We thank the authors for releasing their code, models,
datasets, and benchmarks.

## License

The code is released under the [MIT License](LICENSE). Third-party models, datasets, and simulators remain subject to
their respective licenses.
