from prismatic.vla.datasets.rlds.oxe.configs import (
    LIBERO_DATASETS,
    LIBERO_STANDARD_DATASETS,
    OXE_DATASET_CONFIGS,
)
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

from pathlib import Path


def test_libero_mem_is_registered_without_changing_released_mixture():
    assert "libero_mem" in LIBERO_DATASETS
    assert "libero_mem" in OXE_DATASET_CONFIGS
    assert "libero_mem" in OXE_STANDARDIZATION_TRANSFORMS
    assert OXE_NAMED_MIXTURES["libero_mem"] == [("libero_mem", 1.0)]
    assert OXE_NAMED_MIXTURES["libero_4_task_suites_no_noops"] == [
        (name, 1.0) for name in LIBERO_STANDARD_DATASETS
    ]


def test_libero_mem_launcher_keeps_checkpoint_compatible_contracts():
    launcher = Path(__file__).resolve().parents[1] / "vla-scripts" / "run_libero_mem.sh"
    text = launcher.read_text()
    assert "--vla.data_mix libero_mem" in text
    assert "--vla.action_horizon 20" in text
    assert "--vla.visual_token_pair_offset 31" in text
    assert "--vla.train_ttt_only True" in text
    assert "--vla.ttt_require_full_context True" in text
