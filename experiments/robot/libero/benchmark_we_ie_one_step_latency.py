"""Compare WE and IE one-step inference on one idle H100 PCIe GPU."""

import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.robot.libero.benchmark_predict_action_latency import (
    collect_observations,
    gpu_memory_mib,
    run_group,
)
from experiments.robot.openvla_utils import get_vla
from experiments.robot.robot_utils import set_seed_everywhere


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--we-checkpoint", required=True, type=Path)
    parser.add_argument("--ie-checkpoint", required=True, type=Path)
    parser.add_argument("--base-vlm", required=True, type=Path)
    parser.add_argument("--llm-checkpoint-path", required=True, type=Path)
    parser.add_argument("--vjepa-checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for path in (
        args.we_checkpoint, args.ie_checkpoint, args.base_vlm,
        args.llm_checkpoint_path, args.vjepa_checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required benchmark input missing: {path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if "H100 PCIe" not in gpu:
        raise RuntimeError(f"Table 3 used H100 PCIe; refusing a different GPU: {gpu}")
    if (occupied := gpu_memory_mib()) > 2048:
        raise RuntimeError(f"Allocated H100 is not idle: {occupied} MiB in use before model load")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    set_seed_everywhere(7)
    records = collect_observations(episodes=10, observations_per_episode=23)
    if len(records) != 230:
        raise RuntimeError(f"Expected 230 preloaded observations, got {len(records)}")
    args.output_dir.mkdir(parents=True)
    manifest = [
        {"episode": record.episode, "task_id": record.task_id, "instruction": record.instruction}
        for record in records
    ]
    (args.output_dir / "observation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary = []
    for label, checkpoint, architecture, layer_indices in (
        ("WE_wrapper", args.we_checkpoint, "wrapper", (3, 7, 11, 15)),
        ("IE_inline", args.ie_checkpoint, "inline", tuple(range(16))),
    ):
        config = type("Config", (), {})()
        config.pretrained_checkpoint = checkpoint
        config.base_vlm = args.base_vlm
        config.llm_checkpoint_path = args.llm_checkpoint_path
        config.vjepa_checkpoint_path = args.vjepa_checkpoint_path
        config.levjepa_checkpoint_path = None
        config.num_open_loop_steps = 8
        config.ttt_inference_updates = 1
        model = get_vla(config)
        head = model.action_head
        if not head.ttt_enabled or head.ttt_architecture != architecture or head.ttt_memory_source != "jepa_current":
            raise RuntimeError(f"Wrong TTT architecture/source loaded from {checkpoint}")
        if tuple(head.model.ttt_layer_indices) != layer_indices:
            raise RuntimeError(f"Unexpected TTT block indices for {label}: {head.model.ttt_layer_indices}")
        if head.num_inference_timesteps != 4:
            raise RuntimeError(f"Expected four-step checkpoint default: {checkpoint}")

        print(f"Measuring {label} on {gpu}: {checkpoint.resolve()}", flush=True)
        result = run_group(
            model, records, label=label, steps=1, checkpoint=checkpoint,
            peak_only_measured=True,
        )
        result["peak_memory_scope"] = "process_peak_cuda_allocated_over_200_measured_calls_after_warmup"
        output = args.output_dir / f"{label}_fm1.json"
        with output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        summary.append({
            "model": label,
            "checkpoint": str(checkpoint.resolve()),
            "gpu": gpu,
            "num_inference_timesteps": 1,
            "warmup_calls": 30,
            "measured_calls": 200,
            "expert_mean_ms": result["expert_ms"]["mean"],
            "expert_median_ms": result["expert_ms"]["median"],
            "expert_p95_ms": result["expert_ms"]["p95"],
            "e2e_mean_ms": result["e2e_ms"]["mean"],
            "e2e_median_ms": result["e2e_ms"]["median"],
            "e2e_p95_ms": result["e2e_ms"]["p95"],
            "peak_gpu_memory_mib": result["process_peak_cuda_allocated_mib"],
        })
        print(
            f"{label}: Expert mean={result['expert_ms']['mean']:.3f} ms, "
            f"E2E mean={result['e2e_ms']['mean']:.3f} ms, "
            f"peak allocated={result['process_peak_cuda_allocated_mib']} MiB",
            flush=True,
        )
        del model, head
        gc.collect()
        torch.cuda.empty_cache()

    with (args.output_dir / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved WE/IE comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
