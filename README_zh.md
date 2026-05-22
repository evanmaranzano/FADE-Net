# FADE-Net: 用于轻量级年龄感知的特征融合混合注意力分布估计网络

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Protocol](https://img.shields.io/badge/Protocol-AFAD_72--8--20-informational)

[English](README.md) | [中文文档](README_zh.md)

## 📖 项目概述

**FADE-Net** (前身为 HAL-Net) 是我们轻量级年龄估计系统的**优化迭代版本**。它集成了**多尺度特征融合 (MSFF)**、**空间金字塔池化 (SPP)** 和 **混合注意力机制 (Hybrid Attention)**，面向资源受限场景探索较小参数规模下的年龄估计性能。

**命名 "FADE" 的含义：**
*   **F**eature-fused (特征融合：纹理 + 语义双流)
*   **A**ttention-guided (注意力引导：金字塔坐标注意力)
*   **D**istribution (分布学习：自适应 Sigma DLDL-v2)
*   **E**stimation (估计：鲁棒年龄推理)

**当前内部记录：**
*   **主协议**: AFAD 分层 `72-8-20`，以最终结果文件中的 `Selected_Test_MAE` 作为主表口径
*   **历史最佳单模型**: 测试 MAE 约 **3.057**，仅作历史记录；需按当前 split、TTA 和 checkpoint 元数据重新核验后再写入论文主表
*   **参数量**: 约 **4.84M**
*   **效率验证**: 已有 CPU/GPU smoke benchmark；正式部署结论仍需在目标设备上补测

---

## ✨ 核心特性

1.  **双流架构 (Dual-Stream Architecture)** [新增]: 定义了“纹理分支” (步长-16) 和“语义分支” (步长-32)，同时捕捉细微皱纹和面部轮廓。
2.  **空间金字塔池化 (SPP)** [新增] [6]: 增强的结构设计，通过 SPP 和分层切分进一步提高表征效率。
3.  **混合注意力 (Hybrid Attention)**: 在深层注入 **坐标注意力 (Coordinate Attention, CA) [5]**，在不增加繁重计算的情况下增强空间感知能力。
4.  **DLDL-v2**: 自适应标签分布学习，结合 **排序损失 (Ranking Loss, 0.3)** 和 **加权 L1 损失** [3]。
5.  **鲁棒训练**: **Mixup** + **安全随机擦除 (Safe Random Erasing)** (协同增强) + **标签 Sigma 抖动** 确保特征学习的鲁棒性。
6.  **细粒度增强**: 优化的增强流水线，包含 **仿射变换 (剪切/平移)** 和 **高斯模糊**，用于提升几何和质量鲁棒性。
7.  **预训练**: 使用 **ImageNet1K V2** 权重 (Top-1 75.2%) 进行稳健初始化。

---

## 📂 项目结构

```text
code/
├── src/                  # [源码] 核心逻辑与入口点
│   ├── config.py         # 配置 (超参数, ablation...)
│   ├── model.py          # FADE-Net 架构
│   ├── dataset.py        # 数据集加载与增强
│   ├── train.py          # 主训练脚本
│   ├── web_demo.py       # Web 应用 (Streamlit)
│   ├── gui_demo.py       # GUI 应用 (PyQt5)
│   └── utils.py          # 工具类 (DLDL, EMA, Metrics)
├── scripts/              # [脚本] 工具与预处理
│   ├── preprocess.py     # 数据预处理 (AFAD -> datasets/)
│   ├── plot_results.py   # 可视化
│   └── benchmark_speed.py # 推理速度测试
├── datasets/             # [数据] 预处理后的数据集 (AFAD)
├── docs/                 # [文档] 文档资料
│   └── dataset_setup.md  # 数据集设置指南 
├── runs/                 # [输出] TensorBoard 日志
├── requirements.txt      # 依赖列表
└── README.md             # 项目英文 README
```

---

## 🚀 快速开始

### 1. 环境要求
通过 `requirements.txt` 安装依赖：
```bash
pip install -r requirements.txt
```
*   **核心库**: `torch>=2.0`, `torchvision`
*   **数据处理**: `numpy`, `pandas`, `Pillow`, `opencv-python`
*   **UI/工具**: `streamlit`, `tqdm`, `tensorboard`

### 2. 训练
运行完整训练流程：
```bash
python src/train.py --seed 42 --split 72-8-20 --fresh
```
*   **Checkpoints**: 保存于 `Root Directory`，文件名包含 experiment id
*   **CSV日志**: 保存于 `Root Directory`，文件名包含 experiment id
*   **TensorBoard**: 保存于 `runs/FADE-Net_seed42_...` (自动命名)
*   **覆盖保护**: 同一 experiment id 的日志、checkpoint 和最终结果已存在时，`--fresh` 默认拒绝覆盖；如确需重跑，先归档旧产物、换 `--experiment_tag`，或显式使用 `--overwrite_artifacts`。

### 3. 评估
```bash
python scripts/plot_results.py    # 生成可视化结果
python scripts/benchmark_speed.py     # 测试 FPS
```

---

## 💻 Web 演示
用于实时年龄估计的交互式 Web 界面：
```bash
python -m streamlit run src/web_demo.py

```

## 🖥️ GUI 演示
支持摄像头的本地桌面应用程序：
```bash
python src/gui_demo.py
```


---

## 📊 内部结果与文献参考 (AFAD 数据集)

| 方法 | 骨干网络 | MAE (越低越好 ↓) | 参数量 | 协议说明 |
| :--- | :--- | :--- | :--- | :--- |
| FADE-Net (Ours) | MobileNetV3 | 约 3.057 | 4.84M | 历史内部记录，需按当前 metadata、split 和 TTA 口径复核 |
| GRANET [7] | ResNet-50 | 3.10 | ~25.5M | 文献报告结果，协议可能不同 |
| OR-CNN [2] | VGG-16 | 3.34 | 138M | 文献报告结果，协议可能不同 |
| RAN [9] | ResNet-34 | 3.42 | ~21.8M | 文献报告结果，协议可能不同 |
| CORAL [8] | ResNet-34 | 3.48 | ~21.8M | 文献报告结果，协议可能不同 |
| DEX [1] | VGG-16 | 3.80 | 138M | 文献报告结果，协议可能不同 |

> **说明**: 上表中的外部方法来自公开文献，未在当前仓库中按相同 split、预处理、预训练权重和 TTA 协议重新复现，因此只作为量级参考，不构成严格排名或 SOTA 结论。
>
> **💡 为什么更轻？**  
> 我们移除了冗余的 1000 类 ImageNet 分类头，并将其替换为 **任务特定的 SPP 头**。参数量结论以 `scripts/calc_params.py` 的实际输出为准；精度收益需要在同一协议下通过消融实验验证。

> **注意**: 当前主协议为 AFAD 分层 `72-8-20`，正式论文中必须同时报告 split 文件、split fingerprint、TTA 口径和 seed。

### 📊 与近期 AFAD 专项研究的对比 (2023-2024)
以下只作为文献结果参考，不能直接写成严格优劣排序：

| 方法 | 年份 | 来源 | MAE | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| FADE-Net (Ours) | 2025 | - | 约 3.057 | 历史内部记录，待当前协议复核 |
| **DCN-R34** [11] | 2023 | *ERA Journal* | ~3.13 | 文献报告，协议需核对 |
| **MSDNN** [12] | 2024 | *Electronics* | 3.25 | 文献报告，协议需核对 |
| **ResNet-18** [Baseline] | - | *Standard* | ~3.67 | - |

> **📝 关于性能的说明:** README 主体只使用单模型测试结果作为主口径；集成结果、最佳验证集结果和短训筛选结果应放入补充实验，并明确不参与主表排名。

> **📝 关于分割协议的说明:** 不同论文可能使用不同的数据清洗、年龄范围、训练/测试划分和测试增强。当前仓库使用分层 **72-8-20 分割** (训练/验证/测试)，这只能保证仓库内部实验公平；外部严格比较需要同协议复现。

## 📈 可视化与分析 (Seed 1337)

来自我们表现最好的学术种子 (Seed 1337) 的代表性性能指标。

| **损失收敛** | **MAE 性能** |
| :---: | :---: |
| ![Loss](plots/seed_1337/1_loss_curve.png) | ![MAE](plots/seed_1337/2_mae_curve.png) |
| *训练与验证损失* | *平均绝对误差 (测试集: 3.07)* |

| **学习率调度** | **Batch 稳定性** |
| :---: | :---: |
| ![LR](plots/seed_1337/3_lr_schedule.png) | ![Stability](plots/seed_1337/5_batch_stability.png) |
| *动态 LR 用于调整* | *训练稳定性检查* |

## 🔬 学术严谨性与复现性

为了确保公平比较和科学潜力，我们坚持严格的学术标准：

1.  **固定数据分割**: 数据集划分 (`train`/`val`/`test`) 使用 `seed=42` 生成一次并锁定。所有后续实验均使用此完全相同的分割以保证公平比较。
2.  **多种子训练**: 正式论文建议使用 3 个 seed（42, 3407, 2026）报告 mean±std；历史 seed 结果只在能提供日志、checkpoint 和元数据时纳入主表。
    
    | 种子 (Seed) | 测试集 MAE | 状态 | 说明 |
    | :--- | :--- | :--- | :--- |
    | **1337** | **3.057** | 历史记录 | 最佳单模型记录，需按当前 metadata 复核后引用 |
    | **42** | **3.095** | 历史记录 | 可作为重跑基线 |
    | **2026** | **3.105** | 历史记录 | 可作为重跑基线 |
    | **Mean±Std** | **3.086 ± 0.020** | 历史记录 | 正式论文需重新核对日志与协议 |
    | **Ensemble** | **3.02** | 补充结果 | 不作为主表单模型结果 |
3.  **复现脚本**:
    ```bash
    python src/train.py --seed 42 --split 72-8-20 --fresh
    python src/train.py --seed 3407 --split 72-8-20 --fresh
    python src/train.py --seed 2026 --split 72-8-20 --fresh
    ```

---

## 📚 参考文献

1.  **[DEX]** Rothe R, Timofte R, Van Gool L. DEX: Deep EXpectation of apparent age from a single image[C]//Proceedings of the IEEE International Conference on Computer Vision Workshops. 2015: 10-15.
2.  **[OR-CNN]** Niu Z, Zhou M, Wang L, et al. Ordinal regression with multiple output CNN for age estimation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4920-4928.
3.  **[DLDL]** Gao B B, Xing C, Xie C W, et al. Deep label distribution learning with label ambiguity[J]. IEEE Transactions on Image Processing, 2017, 26(6): 2825-2838.
4.  **[MobileNetV3]** Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1314-1324.
5.  **[CoordAtt]** Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 13713-13722.
6.  **[SPP]** He K, Zhang X, Ren S, et al. Spatial pyramid pooling in deep convolutional networks for visual recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 37(9): 1904-1916.
7.  **[GRA_Net]** Garain A, Ray B, Singh P K, et al. GRA_Net: A deep learning model for classification of age and gender from facial images[J]. IEEE Access, 2021, 9: 85672-85689.
8.  **[CORAL]** Cao W, Mirjalili V, Raschka S. Rank consistent ordinal regression for neural networks with application to age estimation[J]. Pattern Recognition Letters, 2020, 140: 325-331.
9.  **[RAN]** Wang F, Jiang M, Qian C, et al. Residual attention network for image classification[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 3156-3164.
10. **[MobileViT]** Mehta S, Rastegari M. MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer[C]//International Conference on Learning Representations. 2022.
11. **[DCN-R34]** Xi J, Xu Z, Yan Z, et al. Portrait age recognition method based on improved ResNet and deformable convolution[J]. Electronic Research Archive, 2023, 31(11): 6585-6599.
12. **[MSDNN]** Bekhouche S E, Benlamoudi A, Dornaika F, et al. Facial age estimation using multi-stage deep neural networks[J]. Electronics, 2024, 13(16): 3259.
13. **[LDL]** Geng X. Label distribution learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2016, 28(7): 1734-1748.
14. **[Eval-Practice]** Paplham J, Franc V. A Call to Reflect on Evaluation Practices for Age Estimation: Comparative Analysis of the State-of-the-Art and a Unified Benchmark[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 1196-1205.
15. **[APPA-REAL]** Agustsson E, Timofte R, Escalera S, et al. Apparent and real age estimation in still images with deep residual regressors on APPA-REAL database[C]//Proceedings of the 12th IEEE International Conference on Automatic Face and Gesture Recognition. 2017: 87-94.
16. **[AgeDB]** Moschoglou S, Papaioannou A, Sagonas C, et al. AgeDB: The First Manually Collected, In-The-Wild Age Database[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2017: 51-59.
17. **[Ranking-CNN]** Chen S, Zhang C, Dong M, et al. Using Ranking-CNN for age estimation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 5183-5192.
18. **[DRF]** Shen W, Guo Y, Wang Y, et al. Deep regression forests for age estimation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2304-2313.
19. **[MV-Loss]** Pan H, Han H, Shan S, et al. Mean-variance loss for deep age estimation from a face[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 5285-5294.
20. **[Expectation-LDL]** Gao B B, Zhou H Y, Wu J, et al. Age estimation using expectation of label distribution learning[C]//Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence. 2018: 712-718.
21. **[SSR-Net]** Yang T Y, Huang Y H, Lin Y Y, et al. SSR-Net: A compact soft stagewise regression network for age estimation[C]//Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence. 2018: 1078-1084.
22. **[Children-Specialized]** Antipov G, Baccouche M, Berrani S A, et al. Apparent age estimation from face images combining general and children-specialized deep learning models[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2016: 96-104.
23. **[Review-CN]** 张珂, 王新胜, 郭玉荣, 等. 人脸年龄估计的深度学习方法综述[J]. 中国图象图形学报, 2019, 24(8): 1215-1230.


---

## 📜 许可证 (License)
MIT License.
