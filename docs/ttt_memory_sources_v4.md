# 显式与隐式 TTT memory 对照（v4）

四条路线使用相同的 TTT query 和直接残差范围：
`[state, 16 registers, action tokens]`。中间的 JEPA future tokens 不接收
直接 TTT 残差。实验变量只有插入密度和 K/V 来源：

| 架构 | 层 | K/V 来源 |
| --- | --- | --- |
| inline + explicit JEPA K/V | 全部16层 | 当前观测的冻结V-JEPA patch tokens |
| inline + implicit DiT K/V | 全部16层 | 完整post-attention DiT序列 |
| wrapper + explicit JEPA K/V | 3、7、11、15 | 当前观测的冻结V-JEPA patch tokens |
| wrapper + implicit DiT K/V | 3、7、11、15 | 完整post-attention DiT序列 |

完整隐式序列为 `[state, 16 registers, JEPA future tokens, action tokens]`。
隐式路线分别投影 query tokens 与完整 K/V tokens；TTT 输出形状跟 query 一致，
所以残差只加回 state/register/action 的原位置。显式路线继续将 query 投影到
JEPA memory 维度，以外部 representation 作为 K/V。

当前观测显式路线使用 `vla.ttt_memory_source=jepa_current`：当前时刻所有已配置
相机视角先经过冻结的 V-JEPA，归一化后的 patch tokens 直接作为 K/V。预测性
显式路线使用 `vla.ttt_memory_source=jepa`，将 WAM 预测的未来 representation
作为 K/V。当前 Round 4 normal explicit 使用当前观测路线；normal 与 highgpu
则为同一组完整 post-attention DiT K/V implicit 作业竞速，先启动的一方取消
另一分区对应架构的作业。预测性显式路线保留在代码中，但不属于当前竞速作业。

新增配置 `vla.ttt_action_kv_scope=full_dit_tokens`。旧 checkpoint 不含该字段，
加载时默认 `query_tokens`，避免修改既有模型的推理语义。四个新 Slurm 默认使用
`full_dit_tokens`。Round-4 显式运行名标记为
`explicit-current-vjepa-kv`，隐式标记为 `implicit-full-dit-kv`，因此不会把
旧的 future-WAM checkpoint 误当成当前观测 K/V checkpoint 续训。

Round 4 覆盖 128 个物理帧，每隔 8 帧取一个观测，因此每条序列包含 16 次
TTT 更新；segment=8。fast weights 在两个 segment 间保留数值并 detach
梯度，独立序列间重置。
原模型冻结，训练 TTT slow parameters 和 register embeddings。
