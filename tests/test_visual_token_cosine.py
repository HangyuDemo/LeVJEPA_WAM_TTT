from pathlib import Path

import torch

from prismatic.conf.vla import Exp_JEPAVLA_Qvv25_VJEPA_0_5B_LIBERO_90
from prismatic.models.action_heads import VisualTokenCosineHead
from prismatic.models.flow_matching_head.cross_attention_dit import TemporalTTTLayer
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.vlms.prismatic import PrismaticVLM


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_visual_token_cosine_uses_trainable_mlp_and_detached_vjepa_target() -> None:
    torch.manual_seed(7)
    head = VisualTokenCosineHead(d_llm=8, d_target=12)
    llm_visual_tokens = torch.randn(2, 6, 8, requires_grad=True)
    vjepa_target = torch.randn(2, 6, 12, requires_grad=True)

    loss, projected = head(llm_visual_tokens, vjepa_target)
    loss.backward()

    assert projected.shape == (2, 6, 12)
    assert llm_visual_tokens.grad is not None
    assert head.fc1.weight.grad is not None
    assert head.fc2.weight.grad is not None
    assert vjepa_target.grad is None


def test_visual_token_cosine_is_zero_for_identical_normalized_embeddings() -> None:
    torch.manual_seed(11)
    head = VisualTokenCosineHead(d_llm=8, d_target=12)
    llm_visual_tokens = torch.randn(2, 6, 8)
    matching_target = head.align_dimension(llm_visual_tokens).detach()

    loss, _ = head(llm_visual_tokens, matching_target)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_action_memory_uses_last_sequence_tokens_per_sample() -> None:
    hidden = torch.arange(20, dtype=torch.float32).reshape(2, 10, 1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    selected = PrismaticVLM._select_action_memory(hidden, attention_mask, num_action_tokens=3)

    torch.testing.assert_close(selected[0, :, 0], torch.tensor([7.0, 8.0, 9.0]))
    torch.testing.assert_close(selected[1, :, 0], torch.tensor([17.0, 18.0, 19.0]))


def test_action_loss_ignores_padded_chunk_targets(monkeypatch) -> None:
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    head.num_timestep_buckets = 1_000
    head._prepare_state = lambda proprio: proprio
    head.sample_time = lambda batch_size, device, dtype: torch.zeros(batch_size, device=device, dtype=dtype)
    head._predict_velocity = lambda vl_embs, actions, timesteps, state: torch.zeros_like(actions)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )

    action_gt = torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]])
    loss, _ = head(
        torch.zeros(2, 1, 1),
        torch.zeros(2, 1),
        action_gt,
        action_valid_mask=torch.tensor([[True, False], [True, False]]),
    )

    torch.testing.assert_close(loss, torch.tensor(13.0))


def test_temporal_ttt_updates_fast_weights_across_robot_timesteps() -> None:
    torch.manual_seed(17)
    layer = TemporalTTTLayer(dim=8, memory_hidden_dim=16)
    attended_tokens = torch.randn(2 * 3, 4, 8, requires_grad=True)

    memory_out, state = layer(attended_tokens, time_steps=3, tbptt_step_size=2)

    assert memory_out.shape == attended_tokens.shape
    torch.testing.assert_close(state.step, torch.full((2,), 3, dtype=torch.long))
    assert state.fast_weights is not None
    assert all(parameter.shape[0] == 2 for parameter in state.fast_weights.values())

    memory_out.sum().backward()
    assert layer.to_qkv[1].weight.grad is not None
    assert layer.memory_out_layerscale.grad is not None


def test_temporal_ttt_can_read_a_rollout_state_without_writing_it_again() -> None:
    torch.manual_seed(19)
    layer = TemporalTTTLayer(dim=8, memory_hidden_dim=16)
    first_tokens = torch.randn(2, 3, 8)
    _, state = layer(first_tokens, time_steps=1)
    second_tokens = torch.randn(2, 3, 8)

    memory_out, read_state = layer(
        second_tokens,
        time_steps=1,
        prev_fast_weights=state,
        update_memory=False,
    )

    assert memory_out.shape == second_tokens.shape
    assert read_state.fast_weights is state.fast_weights
    torch.testing.assert_close(read_state.step, torch.ones(2, dtype=torch.long))


def test_inference_updates_ttt_on_each_denoising_step() -> None:
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    # predict_action uses the encoder's dtype, even with a mocked predictor.
    head.action_encoder = torch.nn.Linear(1, 1)
    head._prepare_state = lambda proprio: proprio
    head.action_horizon = 2
    head.action_dim = 1
    head.num_inference_timesteps = 4
    head.num_timestep_buckets = 1_000
    head.ttt_enabled = True
    update_flags = []

    def fake_predict_velocity(
        vl_embs,
        actions,
        timesteps_tensor,
        state,
        *,
        memory_tokens=None,
        prev_fast_weights=None,
        return_fast_weights=False,
        update_fast_weights=True,
        **kwargs,
    ):
        update_flags.append(update_fast_weights)
        result = torch.zeros_like(actions)
        return (result, prev_fast_weights) if return_fast_weights else result

    head._predict_velocity = fake_predict_velocity
    actions, _ = head.predict_action(
        torch.zeros(1, 3, 4),
        torch.zeros(1, 1),
        memory_tokens=torch.zeros(1, 3, 4),
        return_fast_weights=True,
    )

    assert actions.shape == (1, 2, 1)
    assert update_flags == [True, True, True, True]


def test_temporal_visual_alignment_ignores_padding_timesteps() -> None:
    head = VisualTokenCosineHead(d_llm=4, d_target=4)
    llm_tokens = torch.randn(1, 2, 3, 4)
    target = head.align_dimension(llm_tokens).detach().clone()
    target[:, 0] = -target[:, 0]

    loss, _ = head(llm_tokens, target, time_valid_mask=torch.tensor([[False, True]]))

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_temporal_action_loss_ignores_padding_timesteps(monkeypatch) -> None:
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    head.num_timestep_buckets = 1_000
    head._prepare_state = lambda proprio: proprio
    head.sample_time = lambda batch_size, device, dtype: torch.zeros(batch_size, device=device, dtype=dtype)
    head._predict_velocity = lambda vl_embs, actions, timesteps, state: torch.zeros_like(actions)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )

    action_gt = torch.tensor([[[[2.0]], [[5.0]]]])
    loss, _ = head(
        torch.zeros(1, 2, 1, 1),
        torch.zeros(1, 2, 1),
        action_gt,
        time_valid_mask=torch.tensor([[True, False]]),
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_public_recipe_keeps_single_visual_cosine_architecture() -> None:
    cfg = Exp_JEPAVLA_Qvv25_VJEPA_0_5B_LIBERO_90()
    script = (REPO_ROOT / "vla-scripts" / "run_visual_cosine_primary.sh").read_text()
    architecture_source = "\n".join(
        path.read_text()
        for path in (
            REPO_ROOT / "prismatic" / "conf" / "vla.py",
            REPO_ROOT / "prismatic" / "models" / "action_heads.py",
            REPO_ROOT / "prismatic" / "models" / "flow_gr00t_action_head.py",
            REPO_ROOT / "prismatic" / "models" / "vlms" / "prismatic.py",
            REPO_ROOT / "prismatic" / "vla" / "datasets" / "datasets.py",
            REPO_ROOT / "prismatic" / "vla" / "materialize.py",
            REPO_ROOT / "prismatic" / "training" / "train.py",
        )
    )

    assert 'RUN_ID_NOTE="visual-cosine-projector-allviews"' in script
    assert 'NPROC_PER_NODE=8' in script
    assert '--nproc-per-node "${NPROC_PER_NODE}"' in script
    assert "--module prismatic.training.train" in script
    assert cfg.expected_world_size == 8
    assert cfg.global_batch_size == 256
    assert cfg.per_device_batch_size == 32
    assert cfg.max_steps == 40_000

    forbidden = (
        "llm_prefix_bidirectional_attention",
        "visual_token_cosine_use_projector_target",
        "visual_token_cosine_layer_idx",
        "visual_token_cosine_projection_type",
        "visual_token_cosine_target_future_only",
        "action_queries",
        "aux_head",
        "vla-lora-last-n-train",
        "vla-vlm-peft-train",
        "DiT-B",
    )
    for option in forbidden:
        assert option not in script
        assert option not in architecture_source


def test_public_vla_config_matches_released_model() -> None:
    cfg = Exp_JEPAVLA_Qvv25_VJEPA_0_5B_LIBERO_90()

    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) == (32, 64, 0.1)
    assert cfg.flow_gr00t_placeholder_tokens > 0
    assert cfg.lambda_visual_token_cosine == 0.5
    assert cfg.visual_token_pair_offset == 31
    assert cfg.d_action == 7
    assert cfg.d_proprio == 8
    assert cfg.action_horizon == 20


def test_removed_experimental_packages_are_not_published() -> None:
    removed_paths = (
        "jepa_wam",
        "lerobot",
        "meta",
        "pretrained_models",
        "experiments/robot/aloha",
        "experiments/robot/server_deploy",
        "prismatic/extern",
        "prismatic/preprocessing",
        "prismatic/models/flow_gr00t_jepa_action_head.py",
        "prismatic/training/train_utils.py",
        "prismatic/util/batching_utils.py",
    )
    for relative_path in removed_paths:
        path = REPO_ROOT / relative_path
        if path.is_dir():
            assert not any(path.rglob("*.py"))
        else:
            assert not path.exists()


def test_vla_scripts_only_expose_training_and_evaluation_launchers() -> None:
    public_scripts = sorted(path.name for path in (REPO_ROOT / "vla-scripts").glob("*.sh"))
    assert public_scripts == [
        "libero_plus.sh",
        "libero_plus_ttt.sh",
        "libero_standard.sh",
        "run_levjepa_wam.sh",
        "run_visual_cosine_primary.sh",
        "ttt_round2_config.sh",
    ]
    assert not (REPO_ROOT / "vla-scripts" / "train.py").exists()


def test_libero_plus_launcher_matches_native_checkpoint_evaluator() -> None:
    launcher = (REPO_ROOT / "vla-scripts" / "libero_plus.sh").read_text()
    evaluator = (REPO_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval.py").read_text()

    for option in (
        "--pretrained_checkpoint",
        "--base_vlm",
        "--llm_checkpoint_path",
        "--vjepa_checkpoint_path",
        "--task_suite_name",
        "--libero_plus_categories",
        "--num_trials_per_task",
        "--save_rollouts",
    ):
        assert option in launcher
        assert option.removeprefix("--") in evaluator

    for removed_option in (
        "--model_family",
        "--action_head_type",
        "--num_images_in_input",
        "--rotation_representation",
        "--use_aux_head",
        "--use_minivlm",
        "--use_wandb",
    ):
        assert removed_option not in launcher


def test_public_launchers_require_explicit_asset_paths() -> None:
    training = (REPO_ROOT / "vla-scripts" / "run_visual_cosine_primary.sh").read_text()
    evaluation = (REPO_ROOT / "vla-scripts" / "libero_plus.sh").read_text()

    for variable in ("LIBERO_DATA", "QVV_PATH", "VJEPA_CKPT", "BASE_VLM_RUN"):
        assert f'${{{variable}:?' in training
    for variable in ("QVV_PATH", "BASE_VLM_RUN", "LIBERO_PATH"):
        assert f'${{{variable}:?' in evaluation
    assert 'VJEPA_CKPT="${VJEPA_CKPT:-}"' in evaluation
    assert 'LEVJEPA_CKPT="${LEVJEPA_CKPT:-}"' in evaluation
    assert "Set VJEPA_CKPT or LEVJEPA_CKPT" in evaluation

    assert "DEFAULT_CHECKPOINT" not in evaluation
    assert "JEPA_ENV" not in training
    assert "JEPA_ENV" not in evaluation
