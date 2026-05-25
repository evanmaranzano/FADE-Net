# FADE-Net: 特征融合混合注意力分布估计网络

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Protocol](https://img.shields.io/badge/Protocol-AFAD_72--8--20-informational)
![Backbone](https://img.shields.io/badge/Backbone-MobileNetV4--Small-brightgreen.svg?style=flat-square)

[English](README.md) | 中文文档

## 项目概述

**FADE-Net**（前身 HAL-Net）是一个面向资源受限场景的轻量级面部年龄估计框架。项目基于 MobileNetV4-Small 骨干网络，提供骨干适配器和多组可配置组件，通过 CLI 标志灵活组合，支持系统性消融实验。

**命名含义：**
- **F**eature-fused -- 纹理 + 语义双流特征融合
- **A**ttention-guided -- 金字塔坐标注意力引导
- **D**istribution -- 自适应 Sigma DLDL-v2 分布学习
- **E**stimation -- 鲁棒年龄推理

**当前协议：** AFAD-Full 数据集，72-8-20 分层划分（formal_v1 tag），目标以 3 seeds（42, 3407, 2026）报告 mean +/- std。主指标为最终结果文件中的 `Selected_Test_MAE`；最终主表仍需完整训练与审计结果支撑。

---

## 模块架构

FADE-Net 由 1 个默认骨干和 9 个主要可配置组件组成，用于受控消融：

| 编号 | 模块 | 缩写 | CLI 标志 | 默认状态 | 说明 |
|:--:|:--|:--|:--|:--:|:--|
| 0 | MobileNetV4-Small 骨干 | -- | `--backbone_name` | 启用 | timm 2024 架构，内置注意力 |
| 1 | 多尺度特征融合 | MSFF | `--no-msff` | 启用 | 纹理 + 语义双流融合 |
| 2 | 空间金字塔池化 v2 | SPP | `--no-spp` | 启用 | 全局-局部融合 |
| 3 | DLDL-v2 标签分布学习 | DLDL | `--no-dldl` | 启用 | 自适应 Sigma |
| 4 | 均值-方差损失 | MV | `--no-mv` | 启用 | 分布约束 |
| 5 | 高频纹理增强分支 | TEX (M1) | `--texture` | 禁用 | Sobel 纹理提取 |
| 6 | 自适应三元组损失 | TRIPLET (M2) | `--triplet` | 禁用 | 动态 margin |
| 7 | 非对称序数损失 | ASYM (M3) | `--asym` | 禁用 | 方向相关权重 |
| 8 | 频域通道注意力 | FREQ (M4) | `--freq` | 禁用 | DCT 频域注意力 |
| 9 | 年龄感知混合专家头 | MOE (M5) | `--moe` | 禁用 | 多专家门控 |

> 注：Hybrid Attention (CoordAtt) 仅对 legacy torchvision 骨干有效；默认 MobileNetV4/timm 路径使用骨干自带注意力，不应把 CoordAtt 计作默认新增模块。

---

## 项目结构

```
FADE-Net/
├── src/                          # 核心源码
│   ├── config.py                 # 配置（骨干适配器、模块开关、超参数）
│   ├── model.py                  # FADE-Net 模型架构
│   ├── train.py                  # 训练循环（实验管理、EMA、AMP、CLI 入口）
│   ├── dataset.py                # AFAD 数据集加载器（DLDL 标签分布、LDS、重试逻辑）
│   ├── utils.py                  # 损失函数（CombinedLoss、EMA、种子管理）
│   ├── backbones.py              # Timm 骨干适配器（MobileNetV4、RepViT）
│   ├── evaluation.py             # 评估工具（predict_probs、TTA）
│   ├── experiment.py             # 实验管理（元数据、fingerprint、审计追踪）
│   ├── ablation_profiles.py      # 消融实验预设配置
│   ├── gui_demo.py               # 桌面 GUI 演示（PyQt5）
│   └── web_demo.py               # Web 演示（Streamlit）
├── scripts/                      # 工具脚本
│   ├── run_backbone_screening.py # 骨干架构筛选
│   ├── audit_paper_results.py    # 论文结果审计
│   ├── plot_results.py           # 训练过程可视化
│   ├── advanced_eval.py          # 高级评估（TTA）
│   ├── swa_average.py            # 随机权重平均
│   ├── calc_params.py            # 参数量 / FLOPs 计算
│   ├── benchmark_speed.py        # 推理速度基准测试
│   └── preprocess.py             # 数据预处理
├── tests/                        # 单元测试（12 个测试文件）
├── docs/                         # 文档
│   ├── ablation_plan_v4.md       # 消融实验计划
│   ├── backbone_screening.md     # 骨干筛选结果
│   ├── paper_core_claims.md      # 论文核心声明
│   └── dataset_setup.md          # 数据集设置指南
├── requirements.txt              # Python 依赖
└── README.md                     # 英文 README
```

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（可选，自动检测）

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：`torch>=2.0`、`torchvision`、`timm>=1.0.0`、`numpy`、`pandas`、`scipy`、`opencv-python`、`Pillow`、`tqdm`、`tensorboard`

### 数据准备

AFAD 数据集放置于 `datasets/AFAD/` 目录（可通过环境变量 `FADE_NET_AFAD_DIR` 或 `--afad_dir` 覆盖）。详见 `docs/dataset_setup.md`。

### 训练

```bash
# 默认配置（MobileNetV4-Small + DLDL + MSFF + SPP + MV）
python src/train.py --seed 42 --split 72-8-20 --split_file_tag formal_v1 --fresh

# 启用全部创新模块
python src/train.py --seed 42 --split 72-8-20 --split_file_tag formal_v1 --fresh --texture --freq --moe --triplet --asym

# 消融实验：禁用特定模块
python src/train.py --seed 42 --split 72-8-20 --split_file_tag formal_v1 --fresh --no-msff --no-spp

# 多种子复现
python src/train.py --seed 42   --split 72-8-20 --split_file_tag formal_v1 --fresh
python src/train.py --seed 3407 --split 72-8-20 --split_file_tag formal_v1 --fresh
python src/train.py --seed 2026 --split 72-8-20 --split_file_tag formal_v1 --fresh
```

主要训练参数：

| 参数 | 默认值 | 说明 |
|:--|:--:|:--|
| `--seed` | 42 | 随机种子 |
| `--split` | 72-8-20 | 数据划分协议 |
| `--epochs` | 120 | 训练轮数 |
| `--batch_size` | 128 | 批大小 |
| `--fresh` | -- | 忽略已有 checkpoint，强制重新训练 |
| `--resume` | -- | 恢复训练（需 checkpoint 元数据匹配） |
| `--experiment_tag` | -- | 附加实验标签（用于 smoke 测试等） |

产物说明：
- Checkpoint 保存于项目根目录，文件名包含实验 ID
- TensorBoard 日志保存于 `runs/` 目录
- 覆盖保护：同一实验 ID 的产物存在时，`--fresh` 默认拒绝覆盖；需使用 `--overwrite_artifacts` 或更换 `--experiment_tag`

### 评估

```bash
# 高级评估（含 TTA：多尺度 0.9x/1.0x/1.1x + 翻转）
python scripts/advanced_eval.py --checkpoint best_model.pth

# 训练可视化
python scripts/plot_results.py

# 推理速度测试
python scripts/benchmark_speed.py

# 参数量 / FLOPs 计算
python scripts/calc_params.py
```

### 骨干筛选

```bash
python scripts/run_backbone_screening.py
```

### 论文结果审计

所有写入论文的结果必须通过审计脚本验证：

```bash
python scripts/audit_paper_results.py
```

### 演示

```bash
# Web 演示
python -m streamlit run src/web_demo.py

# 桌面 GUI 演示
python src/gui_demo.py
```

---

## 核心特性

**可配置组件：** 骨干与非骨干组件通过 CLI 标志控制，支持受控消融实验。实验命名自动根据有效启用模块生成（如 `FADE-Net_DLDL_MSFF_SPP_MV_TEX_FREQ`）。

**实验管理系统：** 基于 metadata fingerprint 的实验追踪，自动检测配置冲突，防止意外覆盖已有结果。

**多种子训练：** 内置 3 个标准种子（42, 3407, 2026），自动报告 mean +/- std。

**测试时增强（TTA）：** 多尺度（0.9x, 1.0x, 1.1x）x 翻转，共 6x 增强。

**指数移动平均（EMA）：** backbone 解冻时自动注册，decay 0.999。

**自动混合精度（AMP）：** 训练全程启用，降低显存占用并加速收敛。

---

## 实验协议

| 项目 | 说明 |
|:--|:--|
| 数据集 | AFAD-Full |
| 划分方式 | 72-8-20 分层划分（formal_v1） |
| 种子 | 42, 3407, 2026（报告 mean +/- std） |
| 主指标 | `Selected_Test_MAE`（来自最终结果文件） |
| TTA | 多尺度（0.9x, 1.0x, 1.1x）x 翻转 |
| 骨干 | MobileNetV4-Small（timm，预训练） |
| 训练轮数 | 120 epochs |
| 优化器 | AdamW（lr=3e-4, wd=4e-4） |
| 调度器 | Cosine Annealing |
| Backbone 冻结 | 前 10 epochs |

---

## 重要说明

1. **当前主协议**：AFAD 72-8-20 分层划分，formal_v1 tag。所有正式结果必须基于此协议。
2. **历史结果**：README 中的历史 MAE（如 ~3.057）来自旧 split/protocol，仅作参考，需按当前协议重跑后才能写入论文主表。
3. **骨干变更**：默认 backbone 已从 MobileNetV3-Large 切换为 MobileNetV4-Small（timm 2024 架构）。Hybrid Attention (CoordAtt) 对 timm 骨干不适用。
4. **结果审计**：所有写入论文的结果行必须通过 `scripts/audit_paper_results.py` 审计，确保 split、seed、TTA 口径和 checkpoint 元数据一致；`paper-ready` 只表示单行证据链通过，最终 mean/std 主表仍需 `scripts/summarize_paper_results.py` 标记为 `complete`。
5. **实验覆盖保护**：同一实验 ID 的产物（checkpoint、日志、最终结果）已存在时，需显式使用 `--overwrite_artifacts` 才能覆盖。

---

## 许可证

MIT License.
