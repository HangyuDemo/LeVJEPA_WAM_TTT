"""Stable episode-level TFDS partitions, applied before any windowing."""


def episode_split_spec(num_episodes: int, validation_percent: int) -> dict:
    if not 0 <= validation_percent < 50:
        raise ValueError("validation_percent must be in [0, 50).")
    held_out = num_episodes * validation_percent // 100
    if validation_percent and (held_out < 1 or held_out >= num_episodes):
        raise ValueError("Dataset is too small for the requested episode holdout.")
    return {
        "num_episodes": num_episodes,
        "validation_episodes": held_out,
        "train_split": f"train[{held_out}:{num_episodes}]" if held_out else "train",
        "validation_split": f"train[:{held_out}]" if held_out else None,
        "method": "TFDS absolute episode slices, validation is canonical prefix; no task stratification",
    }
