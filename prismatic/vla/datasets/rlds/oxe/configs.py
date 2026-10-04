"""RLDS camera and state mappings for the public LIBERO datasets."""

LIBERO_STANDARD_DATASETS = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)
LIBERO_MEM_DATASETS = ("libero_mem",)
LIBERO_DATASETS = LIBERO_STANDARD_DATASETS + LIBERO_MEM_DATASETS

_LIBERO_CONFIG = {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", "gripper_state"],
}

OXE_DATASET_CONFIGS = {name: dict(_LIBERO_CONFIG) for name in LIBERO_DATASETS}
