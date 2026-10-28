### 3.4 训练目标与优化（中文翻译）

我们从同一条 episode 中取出按时间排序的 $T=16$ 个观测，相邻观测间隔 8 个环境帧。每个观测 $i$ 对应一个真实动作块 $\mathbf a_i\in\mathbb R^{H\times d_a}$。采样 $\boldsymbol\epsilon_i\sim\mathcal N(\mathbf0,\mathbf I)$ 和 $u_i\sim\operatorname{Beta}(1.5,1)$，令 $t_i=(0.999-u_i)/0.999$，并构造 $\mathbf x_i^{t_i}=(1-t_i)\boldsymbol\epsilon_i+t_i\mathbf a_i$。目标流速度为 $\mathbf v_i^\star=\mathbf a_i-\boldsymbol\epsilon_i$。动作专家根据当前观测的条件信息 $\mathbf c_i$ 和完成本次记忆写入后的 fast-weight 状态，预测 $\hat{\mathbf v}_i$。设 $m_{ih}\in\{0,1\}$ 表示动作块中第 $h$ 步是否有效，则外层目标为带掩码的 flow-matching 损失：

$$
\mathcal L_{\mathrm{FM}}=
\frac{\sum_{i=1}^{T}\sum_{h=1}^{H}\sum_{d=1}^{d_a}
 m_{ih}(\hat v_{ihd}-v^\star_{ihd})^2}
 {d_a\sum_{i=1}^{T}\sum_{h=1}^{H}m_{ih}}.
$$

在每个选中的 DiT block $\ell$ 中，当前 JEPA 表征提供记忆键 $K_i^{(\ell)}$ 和值 $V_i^{(\ell)}$。将 fast weights 写为相对可学习慢速初始化 $W_\ell$ 的偏移量 $\Delta_i^{(\ell)}$，内层更新为

$$
\begin{aligned}
\ell_{i,\ell}^{\mathrm{TTT}}
 &=\operatorname{MSE}\!\left(f_{W_\ell+\Delta_{i-1}^{(\ell)}}(K_i^{(\ell)}),V_i^{(\ell)}\right),\\
\Delta_i^{(\ell)}
 &=\Delta_{i-1}^{(\ell)}
 -\operatorname{softplus}(\alpha_\ell)g_i^{(\ell)}
 \nabla_{W_\ell+\Delta_{i-1}^{(\ell)}}\ell_{i,\ell}^{\mathrm{TTT}},
\end{aligned}
$$

其中 $g_i^{(\ell)}\in[0,1]$ 是随观测变化的可学习更新门。内层 MSE 用于更新 fast weights，**不**直接加到外层 flow-matching 损失。在每段 8 个观测内部，外层损失的梯度能够穿过可微的 fast-weight 更新；段末保留 fast weights 的数值，但切断其计算图，再传入下一段。每个新采样窗口从 $\Delta_0^{(\ell)}=0$ 开始。

预训练视觉编码器、视觉语言主干与投影器、动作专家中非 TTT 的参数，以及视觉对齐头均冻结。AdamW 只优化 TTT 的慢速参数（包括 $W_\ell$、投影层、门控和内层学习率参数）以及新插入的 register embeddings。训练使用全局 batch size 64、峰值学习率 $2\times10^{-5}$、前 3\% 步线性 warmup、随后余弦衰减至 $10^{-6}$、权重衰减 $10^{-5}$，并将梯度范数裁剪到 1。本组训练不使用未来视觉对齐损失。Flow-matching 训练时，每个观测只采样一个噪声时间点；推理时的去噪步数是独立选择，并非训练时展开的多步目标。
