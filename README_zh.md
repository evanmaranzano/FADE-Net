# FADE-Net

FADE-Net 是面向 AFAD 的轻量级人脸年龄估计模型。当前仓库只保留一条新版训练链路：MobileNetV4-Conv-Small + DCSR + CGBR。

## 当前有效源码

```text
src/
├── train_fade_net.py   # 训练、验证、checkpoint、EMA
├── fade_net.py         # FADE-Net 主模型
├── dcsr_cgbr.py        # DCSR、CGBR、特征适配器和 FADELoss
├── config.py           # 模型与训练配置
├── backbones.py        # timm 骨干适配器
└── experiment.py       # 配置元数据辅助函数
```

旧版 `train.py`、`model.py`、`dataset.py`、`evaluation.py`、`utils.py` 训练栈已经移除。

## 模型与实验协议

- 骨干：timm `mobilenetv4_conv_small`，使用预训练权重
- 年龄范围：15–40 岁，共 26 类
- 输入：RGB 256×256
- 多尺度特征：浅层/中层/深层通道数 32/96/960
- 预测流程：粗年龄分布 → DCSR → 主年龄分布 → CGBR 修正
- 数据集：AFAD
- 划分：identity-disjoint 五折
- 优化器：AdamW，主干和头部使用差分学习率
- 学习率：Cosine Annealing
- EMA：每个 optimizer step 更新，并同步模型 buffer

## 训练

先安装依赖：

```bash
pip install -r requirements.txt
```

单个 fold 示例：

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --split_id 0 \
  --output_dir outputs/fade_net_ema_fix
```

`--split_dir` 中需要存在 `dataset_split_AFAD_15_40_iddisjoint_fold0.json` 至 `fold4.json`。服务器五折启动脚本为 `scripts/train_fade_net.sh`。

## 产物管理

checkpoint 和训练日志属于实验产物，不提交到 Git。论文结果摘要和数据划分元数据属于证据链的一部分时保留在仓库中。

## 相关文档

- `docs/architecture_review.md`：架构与实现审查
- `docs/paper_result_summary.md`：历史结果摘要
- `docs/dataset_setup.md`：AFAD 数据集设置
