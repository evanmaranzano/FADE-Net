# FADE-Net 架构审查与优化方案

## 一、当前架构总览

### 1.1 Backbone

MobileNetV4-Small（timm，2024 架构），预训练 ImageNet-1K，输入 224×224。
- 参数量：约 3.5M
- 输出通道：1024（last_channel）
- 内置注意力机制（ConvNeXt-style），不需要外部 SE 注入

### 1.2 当前启用的模块（8 个）

| 模块 | 代号 | 状态 | 实际效果 |
|------|------|------|----------|
| DLDL-v2 | — | ✅ 启用 | 核心损失，KL + Rank Loss，有效 |
| MSFF | — | ✅ 启用 | 多尺度浅层/中层特征融合，有效 |
| SPP | — | ✅ 启用 | Bottleneck SPP v2，全局局部融合，有效 |
| Texture Branch | M1 | ✅ 启用 | Sobel 纹理增强分支，**冗余** |
| Adaptive Triplet | M2 | ✅ 启用 | 自适应三元组损失，**收益不明确** |
| Asymmetric Ordinal | M3 | ✅ 启用 | 非对称序数损失，**替代 L1，收益不明确** |
| Frequency Attention | M4 | ✅ 启用 | DCT 频域通道注意力，**冗余** |
| MoE Head | M5 | ✅ 启用 | 3 专家混合头，**增加复杂度** |
| Hybrid Attention | HA | ❌ 跳过 | 配置开着但实际未生效（死代码） |
| Mean-Variance | MV | ❌ 未启用 | 配置默认关闭，训练命令未传 --mv |

### 1.3 损失函数组合

```
总损失 = KL + Rank + Triplet + Asym + MoE_Gate
```

- KL：DLDL 分布蒸馏（主损失）
- Rank：CDF 序数回归（λ=0.5）
- Triplet：自适应三元组（λ=0.1）
- Asym：非对称序数（λ=0.1，替代 L1）
- MoE_Gate：专家路由 KL（λ=0.02）

L1 和 Asym 互斥：开了 Asym 后 L1 自动归零。

### 1.4 训练策略

- 120 epoch，前 10 epoch 冻结 backbone
- Cosine Annealing 学习率（3e-4 → 1e-5）
- Mixup（α=0.5，概率 0.5）
- Random Erasing（概率 0.1）
- EMA（衰减 0.999）
- LDS 标签分布平滑
- Hard Distillation（epoch 105+，强制重建 DataLoader）

## 二、问题分析

### 2.1 Hybrid Attention 是死代码

config.py 第 10 行 `use_hybrid_attention = True`，但 model.py 第 232 行：
```python
if not replaced:
    print("[Model] Backbone has no replaceable SE blocks; HA is skipped for this backbone.")
```

MobileNetV4-Small 没有 SE block，CoordAtt 注入直接跳过。配置开着、日志显示 ENABLED，但实际什么都没做。论文里不能把 HA 算进当前结果的归因。

### 2.2 Texture Branch 和 backbone 特征重复

TextureEnhanceBranch 用 Sobel 算子提取灰度纹理，经过 3 层 Conv 降到 64 维。但 MobileNetV4-Small 的浅层特征（stage 1-2）本身就包含丰富的纹理信息，MSFF 已经在融合这些特征了。Sobel 纹理分支和 MSFF 的浅层分支高度重叠，增加 64 维输入但没有提供 backbone 学不到的信息。

### 2.3 Frequency Attention 收益不明确

FrequencyDomainAttention 用 DCT 基提取 3 个频率分量（DC + 水平 + 垂直），生成通道注意力权重。这个设计假设频域能提供通道选择的额外信息，但 MobileNetV4-Small 已经有内置的注意力机制，再加一层频域注意力可能造成梯度冲突。

### 2.4 MoE Head 增加复杂度但没明显提升

3 个专家 + 门控网络，参数量是普通 FC 头的 3 倍以上。MoE 的核心价值是让不同专家专注不同年龄区间，但 AFAD 的年龄分布（15-75）相对集中，不需要这么强的分区能力。门控损失（λ=0.02）也增加了训练不稳定性。

### 2.5 Triplet 和 Asym 同时开可能冲突

Triplet 拉近相似年龄、推开不同年龄，Asym 对低估和高估施加不同惩罚。两者都编码了年龄序关系，但方式不同：Triplet 在 embedding 空间操作，Asym 在预测值空间操作。同时开可能导致梯度方向不一致。

### 2.6 身份泄漏

当前 72/8/20 按年龄分层随机划分，没有考虑身份互斥。AFAD 中同一人可能有不同年龄的多张照片，训练集和测试集可能出现同一人，导致 MAE 偏乐观。CVPR 2024 基准明确指出这是 AFAD 评估的主要问题。

## 三、优化建议

### 方案 A：精简模块 + 身份互斥（推荐）

**移除：**
- Hybrid Attention（死代码，清理配置）
- Texture Branch（和 MSFF 浅层特征重复）
- Frequency Attention（和 backbone 内置注意力冲突）
- MoE Head（增加复杂度但无明显收益）
- Adaptive Triplet（和 Asym 功能重叠）

**保留：**
- DLDL-v2（核心损失）
- MSFF（多尺度融合）
- SPP（全局局部融合）
- Asymmetric Ordinal（替代 L1，编码序关系）
- EMA、LDS、Hard Distillation（训练策略）

**新增：**
- 身份互斥划分（CVPR 2024 协议，已生成）
- Mean-Variance Loss（可选，约束预测分布的均值和方差）

**预期效果：**
- 参数量减少约 30%
- 训练速度提升约 20%
- 过拟合风险降低
- 消融实验更清晰（每个模块都有明确贡献）
- 身份互斥划分提升结果可信度

### 方案 B：方案 A + 更强正则化

在方案 A 基础上：
- 增加 weight_decay 到 1e-3
- 增加 dropout 到 0.4
- 增加 Random Erasing 概率到 0.2
- 考虑 CutMix 替代 Mixup

**适用场景：** 如果方案 A 仍有过拟合。

### 方案 C：更换 backbone

如果 MobileNetV4-Small 的容量不够：
- MobileNetV4-Conv-Medium（约 9.7M 参数）
- EfficientNet-B2（约 9.2M 参数）
- ResNet-50（约 25M 参数，CVPR 2024 基准用这个）

**代价：** 参数量增加，轻量化优势减弱。

## 四、推荐的训练配置

```bash
python src/train.py \
  --seed 42 \
  --split 72-8-20 \
  --split_file_tag iddisjoint \
  --fresh \
  --overwrite_artifacts \
  --afad_dir /data/AFAD \
  --epochs 120 \
  --freeze 10 \
  --asym \
  --mv
```

模块开关：
- DLDL：开（默认）
- MSFF：开（默认）
- SPP：开（默认）
- Asym：开（--asym）
- MV：开（--mv，新增）
- Texture：关（移除 --texture）
- Freq：关（移除 --freq）
- MoE：关（移除 --moe）
- Triplet：关（移除 --triplet）
- HA：关（清理配置）

## 五、预期 MAE 分析

| 配置 | 预期 Test MAE | 依据 |
|------|--------------|------|
| 当前全模块 + 随机划分 | ~3.6（验证集） | 已观测 |
| 精简模块 + 身份互斥 | ~3.3-3.5 | 减少过拟合 + 更严格评估 |
| 精简 + MV + 身份互斥 | ~3.2-3.4 | MV 约束预测分布 |
| 更换 backbone + 精简 | ~3.0-3.2 | 更大容量 |

MAE < 3.2 需要：
1. 身份互斥划分（必须）
2. 精简模块减少干扰（必须）
3. MV Loss 约束分布（建议）
4. 可能需要更大 backbone（如果上述还不够）

## 六、需要确认的问题

1. 是否接受 60/20/20 的身份互斥划分比例？（原计划 72/8/20）
2. 是否保留 Asym 还是改用 L1 + Triplet？
3. 是否尝试 MV Loss？
4. 如果精简后 MAE 仍 > 3.2，是否接受更换 backbone？
