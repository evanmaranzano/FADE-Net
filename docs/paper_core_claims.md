# FADE-Net Paper Claim Boundaries

本文档用于约束 README、论文草稿和实验报告中的核心主张，避免把内部实验写成不可验证的 SOTA 结论。

## 当前可用事实

- 项目当前默认 backbone 为 `timm/mobilenetv4_conv_small`（MobileNetV4-Small，2024 架构）。
- HA（CoordAtt 注入）不适用于 timm backbone（V4 自带注意力机制），消融聚焦 MSFF/SPP/DLDL/MV。
- 当前主实验协议为 AFAD 分层 `72-8-20`，并记录 split file、split fingerprint 与 dataset fingerprint。
- 结果文件应同时报告 `MAE_raw`、`MAE_flip`、`MAE_multi` 和 `Selected_Test_MAE`。
- `Selected_Test_MAE` 是主表口径；TTA 影响应单独列出。
- `mobilenetv4_conv_small` 已完成 `formal_v1` seed42 的 20 epoch 单 seed 训练，并通过论文证据链审计；`mobilenetv4_conv_small_050`、`repvit_m0_9` 仍停留在适配与效率 smoke 检查阶段。
- 当前仍未完成 3 seed 全量训练、原 MobileNetV3 基线同协议复跑、以及候选模型均值/方差主表。

## 可以安全表达的主张

- 在内部 AFAD `72-8-20` 协议下，FADE-Net 展现出较好的精度与参数量平衡。
- FADE-Net 面向年龄估计任务引入了轻量骨干上的任务特定增强，包括注意力注入、多尺度特征融合、SPP 预测头和分布式监督。
- 当前重构引入了 metadata-aware 的训练、恢复、评估和 SWA 流程，降低了混用 split、backbone、TTA 和 loss 配置的风险。
- 现代轻量骨干已接入统一 adapter，可用于比较 MobileNetV4、RepViT 等候选模型。

## 暂时不能写成定论的主张

- 不能写“达到 SOTA”“领先所有轻量级方法”或“超越某论文”，除非外部方法在相同 split、预处理、预训练、TTA 和评估脚本下重新验证。
- 不能把不同论文的 AFAD 数字直接做严格排名；AFAD 划分、年龄范围、清洗方式和测试增强差异会显著影响 MAE。
- 不能把 20 epoch 短训筛选结果写成最终性能。
- 不能把 smoke 级 Params/FLOPs/latency 写成“模型性能提升”，它们只能说明效率和可运行性。
- 不能继续使用“5% 计算成本、90% 性能”这类缺少统一计算口径的比例结论。
- 不能写“端侧设备上的服务器级精度”或“移动端可部署已验证”，除非补充目标设备、batch size、输入尺寸、延迟、内存和功耗等部署证据。
- 不能把未完成的消融表写成实验结论；占位表必须明确标注为模板或计划。

## 推荐贡献表述

1. 提出面向轻量级人脸年龄估计的 FADE-Net 框架，以 MobileNetV4-Small 为骨干，结合多尺度特征融合、空间金字塔池化和分布式监督。
2. 设计纹理-语义双流融合架构（MSFF），显式建模皱纹纹理与脸型结构的双重年龄特征。
3. 引入 Bottleneck SPP 模块捕获多尺度上下文信息，结合 DLDL-v2 标签分布学习和 Mean-Variance Loss 处理年龄标签模糊性。
4. 完善可复现实验协议，记录 experiment id、split fingerprint、dataset fingerprint、backbone 配置、TTA 口径和 loss 分项，避免断点恢复与结果汇总时的配置混淆。

## 论文实验必须补齐

- Baseline 与候选 backbone 的 20 epoch 单 seed 筛选；当前仅 `timm/mobilenetv4_conv_small` seed42 的 `formal_v1` 结果已通过审计。
- Top-2 backbone 与原基线的 3 seed 全量训练。
- 最终主表使用 `--split_file_tag formal_v1` 和带 `_metadata.dataset_fingerprint` 的 split 文件重新跑；旧 split 文件产生的记录只能作为候选筛选或历史参考。
- 模块消融：Baseline、+HA、+DLDL、+MSFF、+SPP、Full，至少 3 seed。
- TTA 消融：raw、flip、multi。
- 效率表：Params、FLOPs、CPU latency、GPU latency，注明硬件和 batch size。
- 若做外部对比，需明确“非统一协议引用结果”或在统一协议下复现实验。

## 结果入表审计

正式论文主表只接受 `scripts/audit_paper_results.py` 判定为 `paper-ready` 的结果。审计会同时检查：

- `final_result_*` 是否含 `MAE_raw`、`MAE_flip`、`MAE_multi`、`Selected_Test_MAE` 和 `Experiment ID`
- `best_model_*` 与 `last_checkpoint_*` 是否为带 metadata 的 packaged checkpoint
- checkpoint metadata 与 split 文件的 `split_fingerprint`、`dataset_fingerprint` 是否一致
- split JSON 是否含 `_metadata`，且未标记 `legacy_upgraded`
- `experiment_tag` 是否为 smoke 类临时实验

示例命令：

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_paper_results.py --split_file_tag formal_v1 --candidates mobilenet_v3_large,mobilenetv4_conv_small,mobilenetv4_conv_small_050 --seeds 42,3407,2026 --output docs/paper_result_audit.csv
```

当前 `docs/paper_result_audit_formal_v1_seed42.csv` 中，`mobilenetv4_conv_small` seed42 的 `formal_v1` 行为 `paper-ready`，可作为单 seed 证据行；它仍不足以支撑最终论文主表结论。`docs/paper_result_audit_current.csv` 是旧审计样例，结果为 `blocked`，只保留为历史参考。

审计后使用 `scripts/summarize_paper_results.py` 生成 `docs/paper_result_summary.md`，只把 `complete` 行作为最终 mean/std 主表证据；`partial` 和 `missing` 行只能作为实验缺口清单。

## 推荐安全摘要句

本文提出 FADE-Net，一种基于 MobileNetV4-Small 的轻量级人脸年龄估计模型。在 AFAD 数据集的固定分层 72-8-20 协议下，模型通过多尺度特征融合、空间金字塔池化和标签分布学习等任务特定模块，在参数量精简的条件下取得了具有竞争力的年龄估计结果。由于不同文献采用的数据划分、预处理和测试协议不同，外部论文结果仅作为参考，不构成严格 SOTA 排名。
