# FADE-Net 消融实验计划（MobileNetV4-Small）

## 背景

主 backbone 已切换为 `timm/mobilenetv4_conv_small`（2024 架构，内置高效注意力）。
HA（CoordAtt 注入）不适用于 timm backbone（无 SE 块可替换）。V4-small 方案聚焦 MSFF、SPP、TEX、FREQ、MOE、TRIPLET、ASYM 等可实际落盘和审计的模块。

## 协议

- **数据集**：AFAD-Full，72-8-20 分层划分
- **Split tag**：`formal_v1`（带 dataset_fingerprint）
- **Seeds**：42, 3407, 2026（3 seeds，用于 mean/std）
- **Epochs**：20
- **Pretrained**：是（ImageNet 预训练权重）
- **审计**：所有结果行必须通过 `scripts/audit_paper_results.py --split_file_tag formal_v1` 判定 `paper-ready`；最终 mean/std 主表只接受 summary 中的 `complete` 行

## 消融配置

| 编号 | 配置名 | DLDL | MV | MSFF | SPP | TEX | FREQ | MOE | TRIPLET | ASYM | 额外 flags |
|------|--------|------|----|------|-----|-----|------|-----|---------|------|------------|
| A0 | Baseline | Y | Y | - | - | - | - | - | - | - | `--no-msff --no-spp` |
| A1 | +MSFF | Y | Y | Y | - | - | - | - | - | - | `--no-spp` |
| A2 | +SPP | Y | Y | - | Y | - | - | - | - | - | `--no-msff` |
| A3 | Full | Y | Y | Y | Y | - | - | - | - | - | none |
| A4 | +TEX | Y | Y | Y | Y | Y | - | - | - | - | `--texture` |
| A5 | +FREQ | Y | Y | Y | Y | - | Y | - | - | - | `--freq` |
| A6 | +MOE | Y | Y | Y | Y | - | - | Y | - | - | `--moe` |
| A7 | +TRIPLET | Y | Y | Y | Y | - | - | - | Y | - | `--triplet` |
| A8 | +ASYM | Y | Y | Y | Y | - | - | - | - | Y | `--asym` |
| A9 | Full+ | Y | Y | Y | Y | Y | Y | Y | Y | Y | `--texture --freq --moe --triplet --asym` |

> **为什么 Baseline 保留 DLDL + MV**：这两项是年龄估计任务的核心 loss 设计（标签分布学习 + 均值方差约束），不是结构模块。消融目的是验证结构模块（MSFF、SPP）的增量贡献。

## 训练命令

### 公共参数

```
AFAD_DIR="F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full"
SPLIT_TAG="formal_v1"
EPOCHS=20
```

### A0: Baseline（V4-small + DLDL + MV，无 MSFF，无 SPP）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp
```

### A1: +MSFF（加多尺度特征融合，无 SPP）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp
```

### A2: +SPP（加空间金字塔池化，无 MSFF）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff
```

### A3: Full（MSFF + SPP + DLDL + MV，全部启用）

> seed42 已有 formal_v1 结果（MAE=3.6130），只需补 seed 3407 和 2026。

```powershell
# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small
```

### A4: +TEX（加高频纹理增强分支）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture
```

### A5: +FREQ（加频域注意力）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --freq

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --freq

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --freq
```

### A6: +MOE（替换为 MoE Head）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --moe

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --moe

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --moe
```

### A7: +TRIPLET（加自适应三元组损失）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --triplet

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --triplet

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --triplet
```

### A8: +ASYM（加非对称序数损失）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --asym

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --asym

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --asym
```

### A9: Full+（全部启用：MSFF + SPP + DLDL + MV + TEX + FREQ + MOE + TRIPLET + ASYM）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture --freq --moe --triplet --asym

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture --freq --moe --triplet --asym

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --texture --freq --moe --triplet --asym
```

## 训练后步骤

### 1. 审计结果

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_paper_results.py --split_file_tag formal_v1 --candidates mobilenetv4_conv_small --seeds 42,3407,2026 --output docs/paper_result_audit_full.csv
```

上面的命令只审计默认 A3/Full 配置。审计 A0-A9 全消融矩阵使用：

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_paper_results.py --split_file_tag formal_v1 --candidates mobilenetv4_conv_small --seeds 42,3407,2026 --ablation_id A0,A1,A2,A3,A4,A5,A6,A7,A8,A9 --output docs/paper_result_audit_full.csv
```

### 2. 生成主表汇总

```powershell
.\.venv\Scripts\python.exe -B scripts/summarize_paper_results.py --audit docs/paper_result_audit_full.csv --candidates A0:timm/mobilenetv4_conv_small,A1:timm/mobilenetv4_conv_small,A2:timm/mobilenetv4_conv_small,A3:timm/mobilenetv4_conv_small,A4:timm/mobilenetv4_conv_small,A5:timm/mobilenetv4_conv_small,A6:timm/mobilenetv4_conv_small,A7:timm/mobilenetv4_conv_small,A8:timm/mobilenetv4_conv_small,A9:timm/mobilenetv4_conv_small --seeds 42,3407,2026 --output docs/paper_result_summary.md
```

### 3. 效率表

```powershell
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A0
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A1
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A2
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A3
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A4
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A5
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A6
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A7
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A8
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small --ablation_id A9
```

> 参数量/FLOPs 脚本若未接入 `--ablation_id`，只能作为默认 A3 参考；正式效率表应以支持 A0-A9 模块开关的脚本为准。

## 预期论文主表结构

| 配置 | Params | FLOPs | MAE (mean +/- std) | TTA |
|------|--------|-------|---------------------|-----|
| A0: Baseline | ? | ? | ? | multi |
| A1: +MSFF | ? | ? | ? | multi |
| A2: +SPP | ? | ? | ? | multi |
| A3: Full | ? | ? | ? | multi |
| A4: +TEX | ? | ? | ? | multi |
| A5: +FREQ | ? | ? | ? | multi |
| A6: +MOE | ? | ? | ? | multi |
| A7: +TRIPLET | ? | ? | ? | multi |
| A8: +ASYM | ? | ? | ? | multi |
| A9: Full+ | ? | ? | ? | multi |

## 当前进度

- [x] A3 Full seed42: MAE=3.6130 (paper-ready single-row evidence; not final mean/std)
- [ ] A3 Full seed3407
- [ ] A3 Full seed2026
- [ ] A0 Baseline seed42
- [ ] A0 Baseline seed3407
- [ ] A0 Baseline seed2026
- [ ] A1 +MSFF seed42
- [ ] A1 +MSFF seed3407
- [ ] A1 +MSFF seed2026
- [ ] A2 +SPP seed42
- [ ] A2 +SPP seed3407
- [ ] A2 +SPP seed2026
- [ ] A4 +TEX seed42
- [ ] A4 +TEX seed3407
- [ ] A4 +TEX seed2026
- [ ] A5 +FREQ seed42
- [ ] A5 +FREQ seed3407
- [ ] A5 +FREQ seed2026
- [ ] A6 +MOE seed42
- [ ] A6 +MOE seed3407
- [ ] A6 +MOE seed2026
- [ ] A7 +TRIPLET seed42
- [ ] A7 +TRIPLET seed3407
- [ ] A7 +TRIPLET seed2026
- [ ] A8 +ASYM seed42
- [ ] A8 +ASYM seed3407
- [ ] A8 +ASYM seed2026
- [ ] A9 Full+ seed42
- [ ] A9 Full+ seed3407
- [ ] A9 Full+ seed2026
- [ ] 效率表
- [ ] 审计汇总
