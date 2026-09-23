# 四路线统一 TTT token 范围（v3）

本页记录旧 v3 配方。当前 full-K/V v4 配方见
[ttt_memory_sources_v4.md](ttt_memory_sources_v4.md)。旧 checkpoint 缺少
`ttt_action_kv_scope` 时仍按本页的 query-token K/V 行为加载。

本次按用户指定，将四条路线统一为：状态＋16 个 register tokens＋action tokens
进入 TTT，并且都接收直接残差。层数、记忆来源和现有训练流程保持各自设置。
这是项目内的范围对齐，不代表完整复现 RoboTTT。

| 路线 | 默认 TTT 插入层（从 0 计数） | K/V 来源 | TTT 查询及直接残差范围 |
| --- | --- | --- | --- |
| inline + JEPA-memory | 全部 16 层 | 显式预测的 JEPA 表征 | 状态＋16 registers＋动作 |
| inline + Action-KV | 全部 16 层 | 选中的状态/register/动作 hidden states | 状态＋16 registers＋动作 |
| wrapper + JEPA-memory | 3、7、11、15 | 显式预测的 JEPA 表征 | 状态＋16 registers＋动作 |
| wrapper + Action-KV | 3、7、11、15 | 选中的状态/register/动作 hidden states | 状态＋16 registers＋动作 |

DiT 序列仍是 `[state, registers, JEPA future tokens, actions]`。每个选中层中，
attention 残差之后选取首部 state/registers 和末尾 actions，送入 TTT，
再将 TTT 输出加回对应位置，之后执行 FFN。中间 JEPA future tokens 不进入
这个 TTT 查询序列，也不接收直接残差，但可通过后续 self-attention 间接受影响。
JEPA 路线传入的显式表征 K/V 与中间的 future tokens 不是同一个接口。
VL conditioning 仍作为 cross-attention 的条件，不直接纳入该 TTT 序列。

Action-KV 是保留的历史名称；在 v3 中它不再表示“只有动作 tokens 产生 K/V”。
两条 Action-KV 路线都用状态、registers、动作产生 Q/K/V；两条 JEPA 路线
用这些 tokens 产生查询，K/V 仍来自 JEPA 表征。

## 参数更新

四个 Slurm 默认仍使用 `train_ttt_only=True`，冻结原有 action DiT、编码器、
语言与视觉骨干等原模型参数。TTT 模块及四条路线的 register embeddings
作为 slow parameters 由优化器更新。此次 inline 的 registers 也会解冻。
Fast weights 是内循环更新并跨 segment 传递的运行状态，不是新增的优化器参数。

32 观测上下文、每 8 观测 detach 并保留 fast weights、独立验证集、action mask、
异常检测、默认 global/per-device batch=1、40000 optimizer steps 均未修改。
跨独立序列仍重置记忆；本次没有改动推理中每次去噪的记忆写入频率。

## 配置、保存与兼容

四个训练入口通过公共脚本默认传入 `TTT_TOKEN_SCOPE=state_register_action`，
保存为 `vla.ttt_token_scope`。在该范围下始终创建 registers，旧
`ttt_wrapper_register_tokens` 开关只决定 legacy wrapper 的行为。
默认运行名包含 `inline-sra16-v3` 或 `wrapper-sra16-v3`，同时保留
`jepa-memory` / `action-kv` 区分；保存目录仍为 `checkpoints/2nd_round`。
不要自行复用旧 RUN_ID；resume 检查会拒绝 token 范围或 register 数变化。

加载 checkpoint 时，缺失 `ttt_token_scope` 的旧配置按 `legacy` 解释，
不自动切换为 v3。这保留本次修改前代码的行为，而不是为所有更早历史版本
自动恢复其当时行为。旧 checkpoint 缺失 wrapper-register 开关时仍默认关闭。
若要显式使用修改前训练范围，设置 `TTT_TOKEN_SCOPE=legacy`。

本次仅修改代码与配置，按要求未执行测试，也没有提交、取消或重启作业。
