# FADE-Net

**由分布反馈驱动的轻量级人脸年龄估计模型。**

[English README](README.md)

FADE-Net 将模型自身的年龄分布状态转化为中间控制信号：**DCSR**（Distribution-Conditioned Scale Routing，分布条件尺度路由）从多层特征中选择当前样本需要的证据，**CGBR**（Correction-Need Guided Bounded Residual Refinement，修正需求引导有界残差细化）通过门控生成不超过 3 岁的残差。FaRL ViT-B/16 只用于训练侧蒸馏，部署时的学生模型不包含教师。

## 已核验实验结果

下表为 AFAD 五个官方主体互斥划分上的测试 MAE。均值 ± 标准差中的标准差是五折总体标准差；数值越低越好。TTA 视图数先在每个 fold 的验证集上，从对称候选 `{2, 4, 6}` 中独立选择，再进行测试集评估。

| 配置 | 参数量 | 每视图 MACs | EMA 1× 测试 MAE | Val 预选 TTA 测试 MAE |
|---|---:|---:|---:|---:|
| **FADE-Net-Small** | **1.576M** | **0.268G** | **3.2042 ± 0.0212** | **3.1585 ± 0.0154** |
| **FADE-Net-Medium** | **7.525M** | **1.114G** | **3.1650 ± 0.0112** | **3.1259 ± 0.0119** |
| Small + Medium 等权概率集成* | 9.101M† | 1.382G† | 3.0687 ± 0.0210 | **3.0503 ± 0.0221** |

**结果应这样解读：**

- 在已完成的五折评估中，Medium 是最强的单模型配置：**5/5 个 fold** 的单视图结果优于 Small，平均改善 0.0392 岁 MAE。
- Small 是面向资源受限场景的方案，参数量 1.576M、每视图 MACs 为 0.268G。
- `3.0503` 是**双模型集成加验证集预选 TTA**的结果，不是单模型部署成绩，应作为更高推理预算下的性能上界单独报告。
- DCSR 与 CGBR 合计只增加 49,882 个参数；其有效性仍应结合仓库中的消融结果和证据边界理解，不能直接扩展为普遍因果结论。

\* 集成在每个 fold 内对 Small 和 Medium 的主年龄分布做等权平均，再聚合预选视图；推理时需要同时部署两个学生模型。

† 两个学生模型在单视图下的组件参数量和 MACs 合计，不是单模型成本；TTA 还会随视图数增加推理开销。

## 评估协议

- **数据集：** AFAD，共 165,501 张图像，实际观测年龄范围为 15–72 岁。
- **输出空间：** 0–80 岁，共 81 类；输出空间更宽不代表 AFAD 含有 15–72 之外的训练标签。
- **划分：** 使用 Paplham 和 Franc 在 CVPR 2024 统一基准中发布的五个主体互斥划分。实验使用的 `AFAD-Full.json` 指纹为 `8813b83131df5e09ccfeb9d513abaa72906da9f816e500dabe7a69e95f086375`。
- **主图像链路：** 使用 AFAD 官方发布的原始裁剪图，不使用统一基准中的 RetinaFace aligned 预处理；与采用 aligned 或外部人脸预训练的结果比较时必须注明协议差异。
- **训练：** ImageNet 预训练的 MobileNetV4-Conv 学生模型、FaRL 分布蒸馏、AdamW、EMA，随机种子为 42。
- **指标：** 平均绝对误差（MAE，单位：岁）。
- **选择纪律：** checkpoint 和 TTA 视图数只使用验证集选择；测试集只用于冻结配置后的报告。训练期 `results.json` 保持测试字段为空，最终测试数字来自独立评估文件。

## 模型结构

```text
输入图像
    │
    ▼
MobileNetV4-Conv 骨干
    ├── 浅层特征 ──┐
    ├── 中层特征 ──┼─► 特征适配器 ─► DCSR ─► 融合特征
    └── 深层特征 ──┘                         │
                                            ▼
                                  主年龄分布 + 期望年龄
                                            │
                                            ▼
                                  CGBR 门控 + 有界残差
                                            │
                                            ▼
                                      最终年龄估计
```

当前有效的学生模型链路为：

```text
src/train_fade_net.py  # 训练、验证、EMA、checkpoint
src/fade_net.py        # FADE-Net 主模型
src/dcsr_cgbr.py       # DCSR、CGBR、特征适配器和 FADELoss
src/backbones.py       # timm / torchvision 特征骨干
src/config.py          # 模型与训练配置
```

最终五折实验使用两种学生骨干：

- `mobilenetv4_conv_small`：1.576M 参数，特征通道 32/96/960。
- `mobilenetv4_conv_medium`：7.525M 参数，特征通道 48/160/960。

独立的 `src/teacher_vit.py` 实现 FaRL ViT-B/16 训练侧教师。每个官方 fold 使用各自的教师 checkpoint，避免跨 fold 身份泄漏；学生推理图中不包含教师。

## 复现单个学生 fold

先安装依赖：

```bash
pip install -r requirements.txt
```

准备数据，但不要将数据集提交到 Git：

```text
datasets/AFAD/                 # AFAD 原始图像目录
data/official/AFAD-Full.json  # 官方基准元数据
```

使用严格官方数据检查，在 Fold 0 训练 Small：

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --official_db data/official/AFAD-Full.json \
  --data_min_age 15 \
  --data_max_age 72 \
  --output_min_age 0 \
  --output_max_age 80 \
  --strict_official_data \
  --split_id 0 \
  --output_dir outputs/fade_net_small_fold0
```

本地有对应 timm 权重时，显式选择 Medium：

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --official_db data/official/AFAD-Full.json \
  --data_min_age 15 \
  --data_max_age 72 \
  --output_min_age 0 \
  --output_max_age 80 \
  --strict_official_data \
  --backbone_source timm \
  --backbone_name mobilenetv4_conv_medium \
  --backbone_weights /path/to/mobilenetv4_conv_medium.e500_r256_in1k-model.safetensors \
  --split_id 0 \
  --output_dir outputs/fade_net_medium_fold0
```

`scripts/run_exp*.sh` 保存了服务器端实际使用的启动命令和单变量实验对照。它们针对原训练服务器的文件系统布局，在新机器上应先适配路径，不要直接盲目执行。

## 教师蒸馏与评估工具

教师在同一官方 fold 上单独训练：

```bash
python src/train_farl_teacher.py \
  --afad_dir datasets/AFAD \
  --official_db data/official/AFAD-Full.json \
  --farl_weights /path/to/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
  --split_id 0 \
  --output_dir outputs/farl_teacher_fold0
```

主要评估和审计入口：

- `scripts/eval_fade_net_tta.py`：使用固定顺序的 1×–6× 视图评估学生 checkpoint。
- `scripts/eval_teacher_tta.py`：按同一视图协议评估 FaRL 教师。
- `scripts/eval_ensemble_tta.py`：评估 Small + Medium 等权概率集成。
- `scripts/summarize_fivefold_results.py`：核验回档的 fold 元数据、划分指纹、TTA 选择和聚合指标。
- `scripts/profile_fade_net_efficiency.py`：测量参数量、MACs 和本机时延。

最终证据摘要见 [`docs/paper/evidence/fivefold_summary.md`](docs/paper/evidence/fivefold_summary.md)，机器可读明细见 [`fivefold_summary.json`](docs/paper/evidence/fivefold_summary.json)。论文初稿和审查说明位于 [`docs/paper/`](docs/paper/)。

## 产物与复现边界

checkpoint、预训练权重、AFAD 图像、运行日志和服务器回档目录有意排除在公开仓库之外，复现实验时需要自行提供。仓库保留结果摘要、划分元数据、评估脚本和论文图表，以便核验报告数字及其选择规则。

现有证据只支持所述 AFAD 协议下的结论，不能单独证明跨数据集泛化、多随机种子不确定性、普遍 SOTA 或移动端实时性能。TTA 和集成结果必须与单视图、单模型部署结果分开报告。

## 仓库文档

- [`docs/paper/evidence/fivefold_summary.md`](docs/paper/evidence/fivefold_summary.md)：五折指标和各 fold 的 TTA 选择。
- [`docs/paper/FADE-Net_中文核心论文初稿.md`](docs/paper/FADE-Net_中文核心论文初稿.md)：中文核心论文初稿。
- [`docs/paper/FADE-Net_深度审查与修改说明.md`](docs/paper/FADE-Net_深度审查与修改说明.md)：证据链和论文表述边界审查。
- [`docs/architecture_review.md`](docs/architecture_review.md)：架构与实现审查。
- [`docs/dataset_setup.md`](docs/dataset_setup.md)：AFAD 数据集设置。
