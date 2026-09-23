# 第二轮：轨迹级验证与数值检测

2026-09-18 新增；不修改模型结构、推理记忆提交规则或训练观测采样间隔。

## 数据划分

四个训练 Slurm 入口默认 `VALIDATION_PERCENT=5`。每个 suite 使用 TFDS 的
固定绝对 episode 切片：验证 `train[:floor(N*5/100)]`，训练使用余下的轨迹。
切片发生在轨迹重复、切窗口和 shuffle 之前，因此不存在同一轨迹的重叠窗口
落入训练和验证两边的问题。选择的是 TFDS 标准顺序的前缀，不是任务分层随机抽样。
四个模型使用同一数据版本和比例时划分完全一致；VALIDATION_SEED 控制验证
噪声，不改变轨迹划分。

本地数据元信息：spatial 432 条（留出21），object 454 条（留出22），
goal 428 条（留出21），10 379 条（留出18）。数字来自当前 dataset_info，
不是对所有 LIBERO 数据版本的通用数量。

每个新 run 写入 `validation-split.json`：TFDS builder/version、精确切片、
episode 数量、dataset_info 的 SHA256、固定验证窗口策略及数量上限。
恢复时检查配置与 manifest；禁止把旧的全数据训练 run 直接改为验证划分后续训。
基础 checkpoint 可能已见过留出轨迹；这是 TTT 微调阶段的 holdout。
动作/proprio 归一化沿用原始全数据统计，保持与官方 checkpoint 的动作尺度兼容；
不是完全隔离基础预训练或归一化统计的全新未知数据基准。

## 固定验证

默认每个 suite 最多16条序列，一条合格 episode 取一个中部的完整 context
窗口。短于 context 的 episode 不参与验证窗口；按固定 TFDS 顺序取前16条
合格轨迹。总计最多64条序列，实际数量记录到日志。没有合格目标则明确报错。
验证数据不 repeat、不 shuffle、不随机重标目标，图像变换不使用随机增强。
单独固定 batch=1，所以训练 GB/PB 改变不会改变验证分组及噪声抽样顺序。

默认 context=32、segment=8；fast weights 在同一个窗口内跨段传递并 detach，
下个窗口重新初始化。模型 eval + 外层 no_grad；TTT 内层梯度更新仍然执行。
不执行外层 backward/optimizer step，不清空训练梯度。完成后恢复各模块原来的
train/eval 状态和 Python/NumPy/PyTorch RNG。不能用 inference_mode 代替 no_grad。
TF 数据管线没有验证期随机操作，也不在验证中重设全局 TensorFlow seed。

默认在启动/恢复时做一次验证，此后每约32000个训练有效观测验证，末步必验证。
GB1/context32为每1000 optimizer steps；GB4/context32为每250步。
验证本身不消耗训练观测预算，不增加 optimizer global_step，但增加实际运行时间。
interval 无法整除时向上取整到完整 optimizer step。

验证计算训练定义的 masked flow-matching velocity MSE，不是动作执行准确率，
也不模拟部署时每观测4次去噪的长 episode rollout。按有效动作 token 数汇总，
总体 loss 是所有 suite 的有效 token 加权平均（不是四个 suite 简单均值）。

JSONL/WandB 指标：

- `Validation/Loss Action`
- `Validation/libero_spatial_no_noops/Loss Action`（object/goal/10 同理）
- `Validation/<suite>/Sequences`
- `Validation/Valid Action Tokens`、`Validation/Sequences`

每个 run 的 `validation-metrics.jsonl` 保留每次验证结果。
训练后验证 loss 创新低时额外保存 checkpoint，并更新
`checkpoints/best-validation-checkpoint.pt` 相对软链接及 `best-validation.json`。
这个最佳只在有验证结果的训练后 checkpoint 之间选择，启动时的 baseline
只记日志，不参与最佳选择。latest 仍正常保存，不把旧模型覆盖为最佳模型。
best_validation_loss 写入 optimizer checkpoint，恢复时一并恢复。

## 数值检测

默认 `check_finite=True`：

- 启动时检查模型参数。
- 每个 TBPTT segment 检查 loss 和全部返回的 fast-weight 张量，非有限即在该段
  backward 之前退出；前面段已经累积的梯度不会被提交给 optimizer。
- 每个 optimizer step 检查裁剪返回的原始梯度范数；NaN/Inf 时不执行 optimizer。
- 每100步、末步、保存前检查可训练参数。
- WandB/JSONL 记录 `Training/Grad Norm`（裁剪前的范数）。

训练 batch 内检测到非有限值时写入 `numerical-failure.json`，包含 step、microbatch、
segment/张量路径；抛出异常停止，不自动跳过坏 batch、不把坏状态保存为 checkpoint。
验证期始终启用 loss/fast-state 检测。初始化验证失败会直接抛异常到 Slurm 日志。
检查不等同于完整 autograd anomaly tracing，未默认启用开销较大的 detect_anomaly。
功能目前限定单GPU、分段的 action-only TTT 配置，多GPU验证会明确拒绝。

## 使用

四个原 Slurm 命令仍可用。默认输出仍为 `checkpoints/2nd_round/`，新 RUN_ID
后缀为 `-val5-n16-s7`，与未启用验证的旧 run 分开。显式 RUN_ID 会覆盖默认名字，
因此提交新实验前应清掉旧 RUN_ID。还没有提交任何新作业，也没有停止已有作业。

```bash
cd /home/ha865618/project/LeVJEPA_WAM_TTT
env -u RUN_ID sbatch run_ttt_action_expert_1gpu.slurm
env -u RUN_ID sbatch run_ttt_action_tokens_1gpu.slurm
env -u RUN_ID sbatch run_ttt_action_expert_inline_1gpu.slurm
env -u RUN_ID sbatch run_ttt_action_tokens_inline_1gpu.slurm
```

可调参数：VALIDATION_PERCENT、VALIDATION_SEQUENCES_PER_SUITE、
VALIDATION_OBSERVATION_INTERVAL、VALIDATION_SEED、PARAMETER_CHECK_INTERVAL。
例如 `VALIDATION_SEQUENCES_PER_SUITE=2 MAX_STEPS=2` 可缩短冒烟验证。
`VALIDATION_PERCENT=0` 关闭划分/验证，但仍保留数值检测，并使用 `val0` 新名字。
保留之前的 WandB 设置方式；新运行的 WandB 名称也带验证标记。

验证入口：

```bash
python -m pytest tests/test_validation.py tests/test_ttt_round2.py tests/test_visual_token_cosine.py -q
```

测试覆盖轨迹切片互斥、窗口选择、四条 TTT 路线验证时的记忆传递/重置、
参数与梯度不变、随机状态/模式恢复、有效动作加权、训练循环验证调度/最佳
checkpoint、非有限状态及梯度中止。

本次验证：38项测试通过；四个Slurm与公共配置通过bash语法检查；训练CLI
参数解析通过。真实四suite数据各读取2条验证窗口并重复读取，动作和图像逐项
一致。当前节点CUDA不可用，尚未执行完整H100/FSDP训练或在线WandB验证。

## 提交前复核补充

再次复核修正：

- 训练指标缓冲接入真实gradient accumulation数量，GB4/PB1时记录四个微批
  loss的平均，而非最后一个微批；指标张量detach，避免日志持有计算图。
- 配置文件保存移到数据manifest校验后，续训另存resume-config.json/yaml，
  保留原config.json/yaml作为训练配方记录。
- step checkpoint与latest均先写同目录临时文件，再原子替换正式文件；序列化
  或latest复制失败时原latest仍完整。SIGKILL可能留下.tmp文件，但不会作为
  latest被读取。latest保持独立普通文件，兼容既有评测工具。

验证结果：43项CPU回归测试通过；另有1项WandB真实SDK离线服务测试通过
（需允许本地socket，禁用联网与API key）。该集成测试默认跳过，显式开启方式：

```bash
RUN_WANDB_OFFLINE_TEST=1 python -m pytest tests/test_training_review.py::test_wandb_offline_tracker_records_validation -q
```

真实默认64条验证序列全部读取成功。spatial/object/goal/10这16条样本各覆盖
5/9/8/9种指令，并未覆盖全部任务，固定小验证集只用于训练诊断/筛选checkpoint。
原先读2条的检查还验证了重复读取的一致性。四个Slurm和公共配置语法通过。
未执行完整GPU训练；未验证在线WandB账户认证和计算节点联网。
