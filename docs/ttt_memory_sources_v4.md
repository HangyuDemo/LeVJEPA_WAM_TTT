# 显式与隐式 TTT memory 对照（v4）

四条路线使用相同的 TTT query 和直接残差范围：
`[state, 16 registers, action tokens]`。中间的 JEPA future tokens 不接收
直接 TTT 残差。实验变量只有插入密度和 K/V 来源：

| 架构 | 层 | K/V 来源 |
| --- | --- | --- |
| inline + explicit JEPA K/V | 全部16层 | WAM预测的JEPA representation |
| inline + implicit DiT K/V | 全部16层 | 完整post-attention DiT序列 |
| wrapper + explicit JEPA K/V | 3、7、11、15 | WAM预测的JEPA representation |
| wrapper + implicit DiT K/V | 3、7、11、15 | 完整post-attention DiT序列 |

完整隐式序列为 `[state, 16 registers, JEPA future tokens, action tokens]`。
隐式路线分别投影 query tokens 与完整 K/V tokens；TTT 输出形状跟 query 一致，
所以残差只加回 state/register/action 的原位置。显式路线继续将 query 投影到
JEPA memory 维度，以外部 representation 作为 K/V。

新增配置 `vla.ttt_action_kv_scope=full_dit_tokens`。旧 checkpoint 不含该字段，
加载时默认 `query_tokens`，避免修改既有模型的推理语义。四个新 Slurm 默认使用
`full_dit_tokens`，运行名包含 `sra16-fullkv-v4`；显式和隐式分别标记为
`explicit-jepa-kv` 与 `implicit-full-dit-kv`，因此不会续训 v2/v3 目录。

训练仍为 context=32、segment=8、GB/PB=1、40000 optimizer steps。
fast weights 在四个 segment 间保留数值并 detach 梯度，独立序列间重置。
原模型冻结，训练 TTT slow parameters 和 register embeddings。
