"""Sequential, single-GPU predict_action latency benchmark on real LIBERO-Plus observations."""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LIBERO_PATH = os.environ.get("LIBERO_PATH")
if not LIBERO_PATH:
    raise RuntimeError("Set LIBERO_PATH to a LIBERO-Plus checkout.")
sys.path.insert(0, LIBERO_PATH)

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import _prepare_observation
from experiments.robot.openvla_utils import get_vla
from experiments.robot.robot_utils import set_seed_everywhere


@dataclass(frozen=True)
class Observation:
    episode: int
    task_id: int
    image: tuple[Image.Image, Image.Image]
    proprio: np.ndarray
    instruction: str


def collect_observations(episodes: int, observations_per_episode: int) -> list[Observation]:
    """Capture short, genuine simulator sequences before any model timing starts."""
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if suite.n_tasks < episodes:
        raise ValueError(f"LIBERO-Plus Spatial has only {suite.n_tasks} tasks; need {episodes}.")
    task_ids = np.linspace(0, suite.n_tasks - 1, episodes, dtype=int).tolist()
    records = []
    for episode, task_id in enumerate(task_ids):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) == 0:
            raise RuntimeError(f"No initial state for Spatial task {task_id}: {task.name}")
        env, instruction = get_libero_env(task, resolution=384)
        try:
            env.seed(7)
            env.reset()
            obs = env.set_init_state(initial_states[0])
            for _ in range(10):
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    raise RuntimeError(f"Spatial task {task_id} ended during observation warmup")
            for index in range(observations_per_episode):
                prepared = _prepare_observation(obs)
                records.append(
                    Observation(
                        episode=episode,
                        task_id=task_id,
                        image=(Image.fromarray(prepared["full_image"]), Image.fromarray(prepared["wrist_image"])),
                        proprio=np.array(prepared["state"], copy=True),
                        instruction=instruction,
                    )
                )
                if index + 1 < observations_per_episode:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    if done:
                        raise RuntimeError(f"Spatial task {task_id} ended before observation {index + 1}")
        finally:
            env.close()
    return records


def gpu_memory_mib() -> int:
    """Device-wide occupancy (including other processes), for the allocated CUDA GPU."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return int((total_bytes - free_bytes) / 1024**2)


def summarize(samples: list[float]) -> dict:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": float(np.percentile(samples, 95)),
    }


def run_group(model, records: list[Observation], *, label: str, steps: int, checkpoint: Path) -> dict:
    set_seed_everywhere(7)
    head = model.action_head
    head.num_inference_timesteps = steps
    head.ttt_inference_updates = 1
    head.ttt_fixed_observation_position = bool(head.ttt_enabled)
    head.ttt_inference_slow_only = False
    assert head.num_inference_timesteps == steps
    assert not head.ttt_enabled or (head.ttt_inference_updates == 1 and head.ttt_fixed_observation_position)

    key = next((key for key in ("libero_spatial", "libero_spatial_no_noops") if key in model.norm_stats), None)
    if key is None:
        raise RuntimeError(f"Spatial normalization stats missing in {checkpoint}: {sorted(model.norm_stats)}")
    expert_samples = []
    original_predict = head.predict_action

    def measured_expert(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = original_predict(*args, **kwargs)
        torch.cuda.synchronize()
        expert_samples.append((time.perf_counter() - start) * 1000)
        return value

    head.predict_action = measured_expert
    e2e_samples = []
    memory_before = gpu_memory_mib()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        prior_episode = None
        for index, record in enumerate(records):
            if record.episode != prior_episode:
                model.reset_ttt_memory()
                prior_episode = record.episode
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions = model.predict_action(
                image=list(record.image),
                instruction=record.instruction,
                unnorm_key=key,
                proprio=record.proprio,
            )
            actions = actions[:8]
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if actions.shape[0] < 8:
                raise RuntimeError(f"{label}: action chunk shorter than 8: {actions.shape}")
            if index >= 30:
                e2e_samples.append(elapsed)
        if len(expert_samples) != len(records):
            raise RuntimeError(f"{label}: expected {len(records)} expert calls, got {len(expert_samples)}")
        expert_samples = expert_samples[30:]
        if len(expert_samples) != 200 or len(e2e_samples) != 200:
            raise RuntimeError(f"{label}: expected 200 samples, got {len(expert_samples)} / {len(e2e_samples)}")
        if not all(np.isfinite(expert_samples)) or not all(np.isfinite(e2e_samples)):
            raise RuntimeError(f"{label}: nonfinite latency")
        if any(e < h for e, h in zip(e2e_samples, expert_samples)):
            raise RuntimeError(f"{label}: E2E < Expert for at least one measurement")
        return {
            "model": label,
            "checkpoint": str(checkpoint.resolve()),
            "num_inference_timesteps": steps,
            "ttt_enabled": bool(head.ttt_enabled),
            "ttt_inference_memory_update": "first_denoise" if head.ttt_enabled else "not_applicable",
            "fixed_observation_memory_position": bool(head.ttt_fixed_observation_position),
            "seed": 7,
            "action_chunk_executed": 8,
            "warmup_calls": 30,
            "measured_calls": 200,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_memory_used_mib_before": memory_before,
            "gpu_memory_used_mib_after": gpu_memory_mib(),
            "process_peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated(0) / 1024**2),
            "expert_ms": {"samples": expert_samples, **summarize(expert_samples)},
            "e2e_ms": {"samples": e2e_samples, **summarize(e2e_samples)},
        }
    finally:
        head.predict_action = original_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--we-checkpoint", required=True, type=Path)
    parser.add_argument("--base-vlm", required=True, type=Path)
    parser.add_argument("--llm-checkpoint-path", required=True, type=Path)
    parser.add_argument("--vjepa-checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.base_checkpoint, args.we_checkpoint, args.base_vlm, args.llm_checkpoint_path, args.vjepa_checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Required benchmark input missing: {path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one allocated CUDA GPU is required, got {torch.cuda.device_count()}")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected H100, got {torch.cuda.get_device_name(0)}")
    occupied_mib = gpu_memory_mib()
    if occupied_mib > 2048:
        raise RuntimeError(
            f"Allocated H100 is not idle before loading models: {occupied_mib} MiB already in use"
        )
    print(f"Allocated H100 idle check: {occupied_mib} MiB used before model load", flush=True)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark output: {args.output_dir}")
    set_seed_everywhere(7)
    records = collect_observations(episodes=10, observations_per_episode=23)
    if len(records) != 230:
        raise RuntimeError(f"Expected 230 preloaded observations, got {len(records)}")
    print(f"Preloaded {len(records)} genuine LIBERO-Plus Spatial observations from 10 short episodes", flush=True)
    args.output_dir.mkdir(parents=True)
    manifest = [{"episode": record.episode, "task_id": record.task_id, "instruction": record.instruction} for record in records]
    (args.output_dir / "observation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for label, checkpoint, expected_ttt in (("original_jepa_wam", args.base_checkpoint, False), ("WE_wrapper", args.we_checkpoint, True)):
        config = type("Config", (), {})()
        config.pretrained_checkpoint = checkpoint
        config.base_vlm = args.base_vlm
        config.llm_checkpoint_path = args.llm_checkpoint_path
        config.vjepa_checkpoint_path = args.vjepa_checkpoint_path
        config.levjepa_checkpoint_path = None
        config.num_open_loop_steps = 8
        config.ttt_inference_updates = 1
        model = get_vla(config)
        if bool(model.action_head.ttt_enabled) != expected_ttt:
            raise RuntimeError(f"Wrong TTT setting loaded from {checkpoint}")
        if expected_ttt and (model.action_head.ttt_architecture != "wrapper" or model.action_head.ttt_memory_source != "jepa_current"):
            raise RuntimeError(f"Wrong WE architecture or memory source in {checkpoint}")
        if model.action_head.num_inference_timesteps != 4:
            raise RuntimeError(f"Checkpoint default denoise steps are not 4: {checkpoint}")
        print(f"Loaded {label}: {checkpoint.resolve()}, default FM steps=4, TTT={expected_ttt}", flush=True)
        for steps in (4, 1):
            print(f"Starting {label}, FM steps={steps}, first_denoise, fixed observation position", flush=True)
            result = run_group(model, records, label=label, steps=steps, checkpoint=checkpoint)
            output = args.output_dir / f"{label}_fm{steps}.json"
            with output.open("x") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            results.append(result)
            print(f"Finished {label}, FM={steps}: Expert={result['expert_ms']['mean']:.3f} ms E2E={result['e2e_ms']['mean']:.3f} ms", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("Model | FM steps | Expert mean (ms) | E2E mean (ms)")
    summary_rows = []
    for result in results:
        print(f"{result['model']} | {result['num_inference_timesteps']} | {result['expert_ms']['mean']:.3f} | {result['e2e_ms']['mean']:.3f}")
        summary_rows.append({
            "model": result["model"],
            "num_inference_timesteps": result["num_inference_timesteps"],
            "expert_mean_ms": result["expert_ms"]["mean"],
            "e2e_mean_ms": result["e2e_ms"]["mean"],
        })
    with (args.output_dir / "summary.json").open("x") as handle:
        json.dump(summary_rows, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
