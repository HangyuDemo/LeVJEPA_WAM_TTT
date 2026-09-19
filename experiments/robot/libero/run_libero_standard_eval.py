"""Evaluate a JEPA-WAM checkpoint on the standard LIBERO benchmark."""

import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

LIBERO_PATH = os.environ.get("LIBERO_PATH")
if not LIBERO_PATH:
    raise RuntimeError("Set LIBERO_PATH to a standard LIBERO checkout before running evaluation.")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import _is_native_prismatic_checkpoint_path
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL.value: 220,
    TaskSuite.LIBERO_OBJECT.value: 280,
    TaskSuite.LIBERO_GOAL.value: 300,
    TaskSuite.LIBERO_10.value: 520,
}


@dataclass
class GenerateConfig:
    pretrained_checkpoint: Union[str, Path] = ""
    base_vlm: Union[str, Path] = ""
    llm_checkpoint_path: Union[str, Path] = ""
    vjepa_checkpoint_path: Union[str, Path] = ""
    levjepa_checkpoint_path: Union[str, Path] = ""
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_trials_per_task: int = 50
    max_tasks: Optional[int] = None
    max_episode_steps: Optional[int] = None
    num_steps_wait: int = 10
    num_open_loop_steps: int = 8
    env_img_res: int = 384
    unnorm_key: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    save_rollouts: bool = False
    seed: int = 7


def _validate_config(cfg: GenerateConfig) -> None:
    if not _is_native_prismatic_checkpoint_path(cfg.pretrained_checkpoint):
        raise ValueError("pretrained_checkpoint must be a local checkpoints/*.pt file.")
    for name in ("base_vlm", "llm_checkpoint_path"):
        path = Path(os.path.expanduser(str(getattr(cfg, name))))
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    vision_paths = (cfg.vjepa_checkpoint_path, cfg.levjepa_checkpoint_path)
    if not any(Path(os.path.expanduser(str(path))).exists() for path in vision_paths if path):
        raise ValueError("Set an existing V-JEPA or LeVJEPA checkpoint path.")
    if cfg.task_suite_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unsupported suite `{cfg.task_suite_name}`; choose from {sorted(TASK_MAX_STEPS)}.")
    if cfg.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if cfg.max_tasks is not None and cfg.max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if cfg.max_episode_steps is not None and cfg.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive when provided.")
    if cfg.num_open_loop_steps <= 0:
        raise ValueError("num_open_loop_steps must be positive.")


def _resolve_unnorm_key(cfg: GenerateConfig, model) -> None:
    candidates = [cfg.unnorm_key, cfg.task_suite_name, f"{cfg.task_suite_name}_no_noops"]
    cfg.unnorm_key = next((key for key in candidates if key and key in model.norm_stats), None)
    if cfg.unnorm_key is None:
        raise ValueError(f"No matching unnorm key in checkpoint statistics: {sorted(model.norm_stats)}")


def _prepare_observation(obs) -> dict:
    proprio = np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    )
    return {
        "full_image": get_libero_image(obs),
        "wrist_image": get_libero_wrist_image(obs),
        "state": proprio,
    }


def _process_action(action: np.ndarray) -> np.ndarray:
    return invert_gripper_action(normalize_gripper_action(action, binarize=True))


def _run_episode(cfg: GenerateConfig, env, task_description: str, model, initial_state):
    env.reset()
    if hasattr(model, "reset_ttt_memory"):
        model.reset_ttt_memory()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []

    episode_steps = cfg.max_episode_steps or TASK_MAX_STEPS[cfg.task_suite_name]
    for step in range(episode_steps + cfg.num_steps_wait):
        if step < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                return True, replay_images
            continue

        replay_images.append(get_libero_image(obs))
        if not action_queue:
            action_queue.extend(get_action(cfg, model, _prepare_observation(obs), task_description))
        action = _process_action(np.asarray(action_queue.popleft()))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, replay_images

    return False, replay_images


def _log(message: str, log_file) -> None:
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def _write_summary(
    cfg: GenerateConfig,
    task_count: int,
    total_episodes: int,
    total_successes: int,
    log_path: Path,
) -> Path:
    """Write evaluation metrics beside any rollout videos."""
    result_dir = Path("./rollout") / cfg.task_suite_name / DATE
    result_dir.mkdir(parents=True, exist_ok=True)
    success_rate = total_successes / total_episodes if total_episodes else 0.0
    summary = {
        "task_suite": cfg.task_suite_name,
        "evaluated_tasks": task_count,
        "trials_per_task": cfg.num_trials_per_task,
        "total_episodes": total_episodes,
        "successes": total_successes,
        "success_rate": success_rate,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "log_file": str(log_path),
        "rollout_directory": str(result_dir),
    }
    json_path = result_dir / f"summary-{DATE_TIME}.json"
    txt_path = result_dir / f"summary-{DATE_TIME}.txt"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(f"Suite: {cfg.task_suite_name}\n")
        handle.write(f"Tasks: {task_count}\n")
        handle.write(f"Trials per task: {cfg.num_trials_per_task}\n")
        handle.write(f"Open-loop actions: {cfg.num_open_loop_steps}\n")
        handle.write(f"Episodes: {total_episodes}\n")
        handle.write(f"Successes: {total_successes}\n")
        handle.write(f"Overall: {total_successes}/{total_episodes} = {success_rate:.4f}\n")
    print(f"Saved evaluation summary at {json_path}")
    print(f"Saved evaluation summary at {txt_path}")
    return json_path


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    _validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    _resolve_unnorm_key(cfg, model)

    log_dir = Path(cfg.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"libero-standard-{cfg.task_suite_name}-{DATE_TIME}.txt"
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    total_episodes = 0
    total_successes = 0
    evaluated_tasks = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        _log(f"Suite: {cfg.task_suite_name}", log_file)
        _log(f"Trials per task: {cfg.num_trials_per_task}", log_file)
        _log(f"Open-loop actions: {cfg.num_open_loop_steps}", log_file)
        _log(f"Tasks in suite: {task_suite.n_tasks}", log_file)

        for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc=cfg.task_suite_name):
            if cfg.max_tasks is not None and task_id >= cfg.max_tasks:
                break

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) < cfg.num_trials_per_task:
                raise ValueError(
                    f"Task `{task.name}` only has {len(initial_states)} initial states, "
                    f"but {cfg.num_trials_per_task} trials were requested."
                )

            env, task_description = get_libero_env(task, resolution=cfg.env_img_res)
            task_successes = 0
            try:
                for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), leave=False):
                    success, replay_images = _run_episode(
                        cfg, env, task_description, model, initial_states[episode_idx]
                    )
                    task_successes += int(success)
                    total_episodes += 1
                    total_successes += int(success)
                    if cfg.save_rollouts:
                        save_rollout_video(
                            replay_images,
                            total_episodes,
                            success,
                            task_description,
                            log_file=log_file,
                            task_suite_name=cfg.task_suite_name,
                        )
            finally:
                env.close()

            _log(
                f"{task_id:02d} {task_description}: {task_successes}/{cfg.num_trials_per_task}",
                log_file,
            )
            evaluated_tasks += 1

        if total_episodes == 0:
            raise ValueError("No standard LIBERO tasks were evaluated.")
        success_rate = total_successes / total_episodes
        _log(f"Overall: {total_successes}/{total_episodes} = {success_rate:.4f}", log_file)

    _write_summary(cfg, evaluated_tasks, total_episodes, total_successes, log_path)
    return success_rate


if __name__ == "__main__":
    eval_libero()
