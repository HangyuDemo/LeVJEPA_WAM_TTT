"""JEPA-WAM LIBERO training mixtures."""

from prismatic.vla.datasets.rlds.oxe.configs import LIBERO_STANDARD_DATASETS

OXE_NAMED_MIXTURES = {
    # Keep the released four-suite recipe unchanged. LIBERO-Mem is a separate
    # benchmark and must not silently enter standard-LIBERO training runs.
    "libero_4_task_suites_no_noops": [(name, 1.0) for name in LIBERO_STANDARD_DATASETS],
    "libero_mem": [("libero_mem", 1.0)],
}
