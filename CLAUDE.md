# FADE-Net

当前项目只维护新版 FADE-Net 训练链路：MobileNetV4-Conv-Small + DCSR + CGBR。

## Active source

```text
src/train_fade_net.py  — 训练/验证/checkpoint/EMA
src/fade_net.py        — 主模型
src/dcsr_cgbr.py       — DCSR、CGBR、FeatureAdapter、FADELoss
src/config.py          — 配置
src/backbones.py       — timm 骨干适配器
src/experiment.py      — 配置元数据辅助函数
```

旧版训练栈已经从 `src/` 移除，不要重新引入 `src/train.py` 作为训练入口。

## Training protocol

- AFAD 官方 `AFAD-Full.json`，实际训练标签 15–72；模型输出空间固定 0–80，共 81 类
- CVPR 2024 official subject-exclusive fold split
- 输入 256×256
- MobileNetV4-Conv-Small pretrained
- backbone lr `3e-5`，head lr `3e-4`
- AdamW，weight decay `5e-4`
- 120 epochs，batch size 64，CosineAnnealingLR
- EMA 每个 optimizer step 更新（衰减 0.999）；参数使用指数平均，buffer 从当前模型同步
- 梯度裁剪 5.0
- CGBR 分阶段启用：epoch 16 开始，epoch 26 完全启用
- 标签分布 σ 2.0；融合通道 96；路由分组 8；残差边界 3.0
- 论文目标：AFAD 五折平均 MAE < 3.20

服务器当前训练命令使用 `src/train_fade_net.py --split_id 0`，输出写入独立实验目录，不覆盖历史 checkpoint。

## Validation

至少执行新版源码语法检查和 EMA 单元检查：

```powershell
F:/Anaconda/envs/fade-net/python.exe -c "from pathlib import Path; compile(Path('src/train_fade_net.py').read_text(encoding='utf-8'), 'src/train_fade_net.py', 'exec')"
```

完整训练结果必须按 fold 和 seed 记录，不能用单个 fold 结果替代论文最终均值和标准差。

## Git and artifacts

- 不提交 `.codegraph/`、`.cursor/`、checkpoint、训练日志和缓存。
- 论文结果文本、数据划分 JSON 和架构文档可提交。
- 提交前检查 `git diff --check`、新版入口语法和训练日志。

## Local overrides

本机特定配置见 `CLAUDE.local.md`（Claude Code 会自动加载；其他工具请主动读取）。
