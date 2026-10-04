#!/usr/bin/env python3
"""Validate the completed LIBERO-Mem TFDS/RLDS dataset before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


EXPECTED_STEP_FIELDS = {
    "observation": {
        "image": ((256, 256, 3), np.uint8),
        "wrist_image": ((256, 256, 3), np.uint8),
        "image_depth": ((256, 256, 1), np.uint8),
        "wrist_image_depth": ((256, 256, 1), np.uint8),
        "image_seg": ((256, 256, 1), np.uint8),
        "wrist_image_seg": ((256, 256, 1), np.uint8),
        "state": ((8,), np.float32),
        "joint_state": ((7,), np.float32),
    },
    "action": ((7,), np.float32),
}


def _check_array(name: str, tensor: tf.Tensor, expected: tuple[tuple[int, ...], np.dtype]) -> None:
    value = tensor.numpy()
    expected_shape, expected_dtype = expected
    if value.shape != expected_shape:
        raise ValueError(f"{name}: expected shape {expected_shape}, got {value.shape}")
    if value.dtype != expected_dtype:
        raise ValueError(f"{name}: expected dtype {expected_dtype}, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ha865618/data/modified_libero_rlds"),
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=961,
        help="Set to 0 to disable the full-dataset episode-count check.",
    )
    args = parser.parse_args()

    version_dir = args.data_root / "libero_mem" / "1.0.0"
    info_path = version_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Incomplete conversion: {info_path} is missing. Wait for download_and_prepare() to finish."
        )

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    builder = tfds.builder_from_directory(version_dir)
    num_episodes = int(builder.info.splits["train"].num_examples)
    if args.expected_episodes and num_episodes != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {num_episodes}")

    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        interleave_block_length=1,
        num_parallel_calls_for_decode=1,
        num_parallel_calls_for_interleave_files=1,
    )
    dataset = builder.as_dataset(split="train", shuffle_files=False, read_config=read_config)
    episode = next(iter(dataset.take(1)))
    step = next(iter(episode["steps"].take(1)))

    for name, expected in EXPECTED_STEP_FIELDS["observation"].items():
        _check_array(f"observation.{name}", step["observation"][name], expected)
    _check_array("action", step["action"], EXPECTED_STEP_FIELDS["action"])

    instruction = step["language_instruction"].numpy().decode("utf-8").strip()
    if not instruction:
        raise ValueError("language_instruction is empty")
    gripper = float(step["action"][-1].numpy())
    if not -1.0001 <= gripper <= 1.0001:
        raise ValueError(f"Raw gripper action is outside [-1, 1]: {gripper}")

    print(f"dataset={builder.info.full_name}")
    print(f"episodes={num_episodes}")
    print(f"first_episode_steps={sum(1 for _ in episode['steps'])}")
    print(f"instruction={instruction}")
    print("LIBERO-Mem RLDS validation passed")


if __name__ == "__main__":
    main()
