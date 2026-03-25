# 摘要

人脸年龄估计是计算机视觉中的典型细粒度回归任务，在智慧零售、安防监控、人机交互、身份认证和数字娱乐等场景中具有较高的应用价值。随着边缘计算与移动端视觉应用的发展，模型不仅需要具备较好的年龄预测精度，还需要在参数规模、推理开销与部署成本之间取得平衡。现有高精度方法通常依赖 VGG、ResNet 或 Transformer 等较重的特征提取网络，虽然在公开数据集上能够获得较低的平均绝对误差，但其参数量和计算量较大，不利于部署在资源受限的终端环境中。围绕这一问题，本文研究一种基于注意力机制的轻量化人脸年龄估计方法。

本文以 MobileNetV3-Large 为基础骨干网络，构建了名为 FADE-Net 的轻量级年龄估计模型。针对年龄线索同时具有局部纹理性与全局结构性的特点，模型在深层阶段采用 Pyramid Attention Injection 策略，将骨干网络最后 4 个含有 SE 结构的模块替换为 Coordinate Attention，以增强空间位置敏感性；在中间层引入纹理与语义双流融合结构，分别从 Block 6 和 Block 12 提取不同尺度的表征，再通过可学习权重实现特征融合；在深层语义分支后加入 Bottleneck Spatial Pyramid Pooling 模块，以较低额外代价引入多尺度上下文；在训练目标上结合标签分布学习、L1 约束、CDF ranking 损失和 Mean-Variance loss，用于缓解年龄标签模糊、序关系弱化和分布过宽等问题。

实验基于预处理后的 AFAD 数据集开展。目录扫描结果表明，实验数据共包含 163373 张图像，年龄目录范围覆盖 15 至 75 岁，其中实际非空样本主要分布在 15 至 72 岁之间。数据划分采用分层 72-8-20 协议，训练集、验证集和测试集样本数分别为 117603、13039 和 32731。根据实验结果文件，FADE-Net 的最佳单模型在 seed1337 下取得 3.0574 的测试集 MAE；seed42 与 seed2026 的结果分别为 3.0951 和 3.1047，三次实验均值为 3.0857，标准差为 0.0204。模型总参数量经实际计数为 4.8415M，可记为 4.84M。结果说明该方法在保持较低参数规模的同时，能够获得具有竞争力的年龄估计性能。

本文的工作价值主要体现在两个方面：一方面，从结构设计上验证了轻量骨干网络与注意力、多尺度融合、标签分布学习相结合的有效性；另一方面，从工程实现角度给出了较完整的训练、验证与测试流程，为进一步扩展可视化分析与补充对比实验提供了稳定基础。

关键词：人脸年龄估计；轻量化网络；注意力机制；标签分布学习；深度学习

# Abstract

Lightweight face age estimation is an important research topic in computer vision because practical applications often require both prediction accuracy and efficient deployment. Existing high-performance solutions usually depend on heavy backbones such as VGG, ResNet, or Transformer-based models. Although these methods can obtain low mean absolute error on public benchmarks, their parameter size and computational cost make them less suitable for mobile or edge-side deployment. To address this issue, this study investigates a lightweight age estimation framework with task-oriented structural refinement.

The proposed model, named FADE-Net, is built on MobileNetV3-Large and introduces several task-oriented improvements. First, a Pyramid Attention Injection strategy replaces the last four squeeze-and-excitation modules in the deep layers with Coordinate Attention, improving spatial sensitivity to age-related facial regions. Second, a texture-semantic dual-stream fusion branch is designed to capture both local wrinkle patterns and global facial structure from intermediate layers. Third, a bottleneck spatial pyramid pooling module is inserted before the prediction head to aggregate multi-scale contextual information with limited extra parameters. In addition, the training objective combines label distribution learning, L1 supervision, CDF ranking loss, and mean-variance loss to better model age ambiguity and ordinal relationships.

Experiments are conducted on the processed AFAD dataset. The scanned dataset contains 163373 images, with directory labels ranging from 15 to 75 years old and effective non-empty samples mainly covering 15 to 72 years old. A stratified 72-8-20 protocol is adopted, producing 117603 training samples, 13039 validation samples, and 32731 test samples. According to the result files, the best single model achieves a test MAE of 3.0574 with seed1337, while seed42 and seed2026 obtain 3.0951 and 3.1047 respectively. The multi-seed mean result is 3.0857 with a standard deviation of 0.0204. The actual parameter count of FADE-Net is 4.8415 million, showing that the proposed method maintains a good balance between accuracy and lightweight design.

The results indicate that a carefully designed lightweight backbone, combined with attention injection, multi-scale fusion and distribution-aware supervision, can provide a feasible solution for efficient face age estimation in resource-constrained scenarios.

Keywords: face age estimation; lightweight network; attention mechanism; label distribution learning; deep learning

# 目录

# 1 绪论

## 1.1 研究背景与意义

人脸年龄估计是指根据单张人脸图像或视频帧中的面部信息自动预测个体年龄的任务。从应用角度看，这项技术可以服务于多个现实场景。例如，在智慧零售中，年龄估计可辅助用户画像构建与客群分析；在安防与门禁系统中，年龄信息可作为辅助属性提升检索与筛查效率；在社交娱乐和数字内容生成场景中，年龄估计可支持滤镜、虚拟形象生成和内容个性化推荐；在医疗健康与人口统计研究中，年龄相关表征还可为风险分析和群体结构评估提供基础特征。随着摄像头设备和边缘视觉终端的大规模普及，年龄估计系统逐步从实验室环境走向真实场景，其性能评估标准也从单一的精度竞争转向精度、速度、参数规模和可部署性的综合平衡。

然而，人脸年龄估计并不是一个容易的问题。首先，年龄变化在视觉上具有连续性和模糊性，相邻年龄之间的差异往往较小，而不同个体之间还受到遗传、妆容、光照、姿态、表情、拍摄质量等多重因素干扰。其次，年龄线索既体现在皱纹、皮肤纹理、法令纹等局部细节，也体现在脸型轮廓、五官比例和骨骼结构等较高层次的语义信息上，单一尺度特征很难充分表达这些变化。再次，许多实际系统需要在普通 GPU、便携式工控机甚至 CPU 环境中运行，过大的模型会带来显著的存储与推理负担。因此，如何在轻量化前提下保留关键年龄特征，是一个具有理论意义和工程价值的问题。

从学术研究的角度看，早期年龄估计主要依赖人工设计特征，例如局部二值模式、Gabor 特征和多尺度纹理描述子等。这类方法实现简单，但泛化能力有限，面对复杂姿态和跨域数据时效果不稳定。近年来深度学习方法逐渐成为主流，卷积神经网络通过端到端特征学习显著提升了年龄估计性能；但与此同时，参数膨胀、训练成本高、部署难等问题也日益突出。因而，围绕轻量化架构、注意力增强、多尺度融合和标签分布学习展开研究，不仅能够改进模型精度，也能够提升模型在边缘设备上的实际应用价值。

综合以上因素，研究基于注意力机制的轻量化人脸年龄估计方法具有较强的现实必要性。它既回应了移动端部署的工程需求，也能够为理解年龄线索的层次性、多样性和模糊性提供新的实现路径。

## 1.2 国内外研究现状

现有年龄估计方法大致可以分为回归方法、分类方法和标签分布学习方法三类。回归方法将年龄直接视作连续值进行预测，建模思路清晰，损失函数通常采用 L1 或 L2 形式，但这种做法容易忽略年龄标签之间的邻近关系。分类方法将每一个年龄视为一个独立类别，通过 softmax 进行预测，再利用期望或后处理恢复年龄值。该策略能够利用成熟的分类框架，但不同年龄类别之间的序关系表达不充分。标签分布学习则进一步考虑年龄标签本身具有模糊边界的特点，将单一年龄扩展为围绕真实年龄分布的概率向量，从而提升模型对邻近年龄的容忍度，在年龄估计研究中得到了广泛关注[3]。

在网络结构方面，DEX 采用 VGG-16 进行 apparent age estimation，是深度学习应用于年龄估计的代表性工作之一[1]。随后，OR-CNN、CORAL 等方法从序回归角度建模年龄标签之间的有序关系，推动了年龄估计从普通分类向 ordinal learning 方向演进[2][8]。与此同时，Residual Attention、GRANET 等模型尝试将注意力机制融入年龄估计任务，以增强对关键区域的感知能力[7][9]。这些方法在精度上有明显提升，但大多依赖较重的主干网络，难以在轻量部署场景中直接应用。

近年来，轻量化视觉网络的快速发展为年龄估计任务提供了新的可能。MobileNet 系列通过深度可分离卷积降低计算开销，ShuffleNet 通过通道混洗增强跨通道信息交互，EfficientNet 和 MobileViT 则在结构搜索与轻量 Transformer 方向上探索了更优的精度效率平衡[4][10]。其中，MobileNetV3 兼顾了神经结构搜索与人工设计经验，在多种移动视觉任务中都表现出较强的实用性。但将通用轻量网络直接迁移至年龄估计任务时，往往仍会出现表征不足、纹理敏感性不强和对局部结构关注不够的问题。

注意力机制也是当前年龄估计研究中的重要方向。SE 模块通过显式建模通道依赖提高特征选择性，但不直接保留空间位置编码；CBAM 在通道与空间两个维度进行重标定；Coordinate Attention 则在轻量结构中融入位置编码，使模型在保持较低复杂度的同时获得更强的方向感知能力[5]。对于年龄估计而言，面部不同区域的年龄信息贡献并不一致，例如额头、眼角、嘴角和下颌区域均携带不同类型的年龄线索，因此空间敏感的注意力模块具有较高的应用潜力。

除网络本身之外，训练策略也是影响年龄估计性能的重要因素。标签分布学习通过高斯分布或自适应分布刻画年龄标签的模糊性，排序损失能够加强年龄的有序关系约束，Mean-Variance loss 则从分布统计角度约束模型输出的均值和方差。综合来看，当前研究的主线已经从单纯追求更深的网络，逐渐转向更符合年龄估计任务特性的结构设计和损失建模。本文的方法设计即围绕轻量骨干、注意力注入、多尺度融合与分布学习展开。

## 1.3 研究内容与主要工作

本文的研究目标是在保证模型轻量化的前提下，提高人脸年龄估计的准确性与稳定性。围绕这一目标，本文结合模型实现与实验结果，对结构设计、训练策略和性能分析进行了系统梳理，主要工作可以概括为以下几个方面。

第一，基于 MobileNetV3-Large 构建轻量骨干网络。相较于 ResNet、VGG 等较重主干，MobileNetV3 在参数量与计算量方面更适合部署场景，但直接用于年龄估计时可能对局部纹理和空间位置的建模不足。为此，本文没有简单沿用原始分类头，而是以骨干特征为基础设计更贴合年龄估计任务的预测结构。

第二，提出 Pyramid Attention Injection 策略。该策略保留浅层的原始 SE 结构，仅在骨干网络后部最后 4 个含有 SE 的模块中引入 Coordinate Attention。这样做的动机在于：浅层主要处理边缘和基础纹理，使用原始结构可以节省计算；深层承载更强语义信息，对面部区域位置更敏感，更适合采用含位置信息的轻量注意力模块。该策略体现了“按层注入”的设计思想，有助于在额外开销有限的前提下提升有效表征。

第三，设计纹理与语义双流融合结构。年龄线索既依赖高分辨率纹理，也依赖低分辨率语义结构，因此本文从 Block 6 和 Block 12 分别提取纹理分支和语义分支，通过 1×1 卷积统一通道后进行加权融合，再与深层语义向量拼接，从而兼顾局部细节与整体结构信息。

第四，引入 Bottleneck SPP 与分布学习损失组合。Bottleneck SPP 使用 5×5、9×9 和 13×13 的最大池化分支捕获多尺度上下文；损失函数则以 KL 散度为主，同时加入 L1、CDF ranking 和 Mean-Variance loss，用于刻画年龄标签的模糊性、序关系以及预测分布的集中性。

第五，结合真实训练与测试结果，对模型进行实验分析。本文统一采用 AFAD 分层 72-8-20 口径，引用实验结果文件中的 MAE 数据，并对多 seed 结果、参数规模和训练曲线进行整理，以验证模型在精度与轻量化之间的平衡效果。

## 1.4 论文结构安排

本文全文共分为五章。第一章为绪论，介绍研究背景、相关研究进展、研究目标与主要工作。第二章介绍人脸年龄估计相关理论，包括轻量化网络、注意力机制、标签分布学习及多尺度特征融合等基础内容。第三章详细阐述 FADE-Net 的结构设计与关键实现，包括骨干网络、注意力注入、双流融合、SPP 模块、损失函数和训练推理策略。第四章给出实验设置、数据口径、结果表格与分析讨论。第五章总结全文并对后续研究方向进行展望。

# 2 相关理论与研究基础

## 2.1 人脸年龄估计任务定义

从问题形式看，人脸年龄估计属于细粒度视觉理解任务，其输出既可以是离散年龄标签，也可以是连续年龄值。与性别分类或人脸识别不同，年龄估计的标签之间具有天然的有序性和连续性。例如，25 岁与 26 岁的视觉差异往往远小于 25 岁与 55 岁之间的差异，因此模型既要学习不同年龄段的可分性，也要尊重相邻年龄之间的相似性。此外，同一真实年龄在不同个体上的面部表现也存在显著差异，导致年龄标签具有较强的不确定性。

在评价指标方面，平均绝对误差 MAE 是年龄估计最常用的指标之一。它直接统计预测年龄与真实年龄之间的绝对差值平均值，表达清晰、易于跨方法比较。对于标签分布学习模型，通常先输出每个年龄类别的概率分布，再通过期望计算得到最终预测年龄。因此，实验部分以 MAE 作为主要评价标准，同时辅以多 seed 稳定性分析。

## 2.2 轻量化卷积网络

轻量化视觉网络的核心目标是在减少参数量和计算量的同时，尽量保留表征能力。MobileNetV3 是典型的轻量骨干，其设计融合了深度可分离卷积、倒残差结构、SE 注意力以及基于神经架构搜索的模块组合[4]。相较于传统卷积网络，深度可分离卷积将标准卷积分解为逐通道卷积和逐点卷积，显著降低了乘加运算量；倒残差结构则通过先扩展后压缩的方式提高特征表达效率。

对年龄估计任务而言，轻量骨干的优点在于可部署性，但局限在于通用分类网络的高层特征更偏向类别判别，而年龄估计需要兼顾纹理细节和连续年龄变化。因此，在轻量网络基础上加入任务定制的结构增强，是提升性能的关键路径。本文选择 MobileNetV3-Large 作为骨干，正是考虑到其成熟、稳定且具备较好的工程基础，同时也为引入注意力和多尺度模块提供了合适的结构载体。

## 2.3 注意力机制

注意力机制的基本思想是根据任务需求动态强调关键特征并抑制无关特征。SE 模块通过全局池化和通道重标定学习通道权重，在轻量模型中广泛应用，但其主要建模通道依赖，对精确空间位置的刻画有限。CBAM 进一步在空间维度增加显式权重分配，但额外开销和结构复杂度相对更高。Coordinate Attention 通过分别沿高和宽方向聚合信息，再将位置编码注入通道权重，在保持轻量的同时兼顾了方向性和位置敏感性[5]。

对于年龄估计任务，注意力机制的作用主要体现在两个方面。其一，不同区域的年龄线索贡献不同，模型应更加关注眼周、额头、法令纹等区域；其二，面部年龄变化不仅是纹理变化，还与面部结构位置分布相关。基于此，本文在深层阶段引入 Coordinate Attention，希望在不显著增加参数量的前提下增强模型对关键区域的响应能力。

## 2.4 标签分布学习与序关系建模

标签分布学习的核心思想是将单点标签扩展为一个分布，以描述标签的不确定性[3]。在人脸年龄估计中，这种思路尤其自然，因为一个人的真实年龄虽然唯一，但视觉年龄存在主观波动，相邻年龄之间往往难以严格分界。若模型只以单点监督训练，容易对标签过拟合；而若将真实年龄周围的邻近年龄给予一定概率质量，则更符合任务本身的连续性特征。

除分布建模外，序关系约束也是年龄估计的重要问题。年龄标签之间存在天然的“大小关系”，如果模型预测结果打破这种顺序，往往说明其内部表征并不稳定。本文采用的 ranking 损失并非重新设计二元序分类头，而是通过预测分布的累积分布函数与目标分布进行比较，使用二元交叉熵形式约束其整体形状。这种做法保留了现有 softmax 输出结构，同时在一定程度上补足了年龄序信息。

## 2.5 多尺度特征融合与上下文建模

年龄线索具有明显的多尺度特性。高分辨率特征有利于捕获皱纹、斑点、毛孔等局部纹理，而低分辨率语义特征更适合表达脸型、下颌轮廓、额头宽度等全局结构信息。如果只依赖高层特征，模型可能忽视细节；如果只依赖浅层特征，又难以获得稳定语义。因此，多尺度特征融合成为提升年龄估计性能的重要策略。

Spatial Pyramid Pooling 通过多个不同感受野的池化分支融合上下文信息，是一种经典而有效的多尺度设计[6]。相比直接堆叠更深卷积层，SPP 在参数增长较小的情况下即可扩大感受野。本文在深层特征后加入 Bottleneck SPP，并配合中层双流融合结构，形成“中层纹理语义融合 + 深层上下文增强”的整体方案，为轻量骨干补足了任务所需的细粒度信息。

# 3 FADE-Net 模型设计与关键实现

## 3.1 模型总体结构

本文采用的 FADE-Net 以 MobileNetV3-Large 为骨干网络，并在此基础上加入注意力注入、多尺度融合和分布学习损失。整体流程可以概括为：输入 224×224 的对齐人脸图像后，首先由骨干网络提取层级特征；随后从中间层提取纹理分支与语义分支进行融合；深层语义特征经过 Bottleneck SPP 进一步聚合上下文；最后将深层语义向量与双流融合向量拼接，经两层全连接预测 81 维年龄分布，再通过期望运算得到最终年龄值。

从代码实现看，模型的类别空间设置为 81 维，对应 0 至 80 的离散年龄索引。虽然当前实验数据的有效年龄样本主要集中在 15 至 72 岁之间，但统一的输出空间便于使用标签分布学习、排序约束和期望回归进行建模，也便于迁移到其他覆盖更广年龄范围的数据集。

图 3-1 展示了依据 `src/model.py` 中模型实现归纳得到的 FADE-Net 网络结构，给出了输入、骨干、双流分支、注意力注入、SPP、预测头以及期望回归输出之间的关系。

![图 3-1 FADE-Net 网络结构示意图](output/doc/assets/fade_architecture.png)

为便于将文中描述与实现细节逐项对应，表 3-1 对模型的关键组成部分进行了归纳。表中模块名称、通道数和层级选择均来自当前实现的默认配置。

| 组件 | 代码实现位置 | 关键配置 | 作用 |
|---|---|---|---|
| 骨干网络 | `LightweightAgeEstimator.backbone` | MobileNetV3-Large + ImageNet1K_V2 | 提取基础视觉特征 |
| 混合注意力 | 最后 4 个 SE 模块替换为 CoordAtt | reduction=16 | 增强深层空间感知 |
| 双流融合 | Block 6 + Block 12 | 40ch/112ch -> 64ch -> 128ch | 融合纹理与语义特征 |
| Bottleneck SPP | `BottleneckSPP` | 5/9/13 池化 + 1×1 压缩到 512ch | 聚合多尺度上下文 |
| 预测头 | `final_head` | Linear -> 1024 -> 81，dropout=0.35 | 输出年龄分布 |

表 3-1 FADE-Net 关键模块与默认配置

## 3.2 基于层次注入的混合注意力设计

原始 MobileNetV3-Large 已在部分倒残差模块中使用 SE 结构，其优势是计算代价低、易于集成，但不足在于对空间位置信息表达较弱。对于年龄估计任务而言，纹理和结构线索并非在整张人脸上均匀分布，而是具有明确的局部区域差异。为此，本文并未采用全网络统一替换的方式，而是通过遍历骨干网络模块结构，定位带有 SqueezeExcitation 的层，并仅替换最后 4 个目标模块。

这种设计体现出明显的层次化思想。浅层特征主要对应边缘、颜色和简单纹理，替换注意力结构的收益有限，且容易引入不必要的开销；深层特征更接近面部语义表达，需要模型区分不同区域的贡献度，因此更适合使用 Coordinate Attention。具体实现中，CoordAtt 先分别沿高度与宽度方向进行自适应平均池化，再将两个方向的特征拼接，通过 1×1 卷积、归一化和激活函数生成方向相关的权重，最后对输入特征进行逐元素重标定。

若记输入特征为 $X \in \mathbb{R}^{C \times H \times W}$，则模型分别生成沿两个方向聚合的描述子：

$$
y_h(c, h) = \frac{1}{W}\sum_{i=1}^{W} X(c, h, i)
$$

$$
y_w(c, w) = \frac{1}{H}\sum_{j=1}^{H} X(c, j, w)
$$

随后再通过共享变换和分支卷积恢复为两个方向的注意力权重。由于该模块主要使用 1×1 卷积和方向池化，因此额外参数较少，适合与轻量化骨干结合。

## 3.3 纹理与语义双流融合

人脸年龄的视觉表现具有明显的双重属性。一方面，皱纹、肤质粗糙度、细纹和斑点等局部细节通常出现在较高分辨率特征中；另一方面，脸型、下颌轮廓、眼窝深浅以及五官比例等全局形态更适合由较深层语义特征编码。若仅使用最后一层特征，很容易忽略对高分辨率细节的利用。针对这一问题，本文从 MobileNetV3-Large 的 Block 6 与 Block 12 分别截取特征图，构建纹理分支和语义分支。

在代码实现中，Block 6 的输出分辨率约为 28×28，通道数为 40；Block 12 的输出分辨率约为 14×14，通道数为 112。两个分支先分别通过 1×1 卷积投影到 64 个通道，再将纹理分支自适应平均池化到 14×14，与语义分支保持空间对齐。随后，模型设置一个长度为 2 的可学习参数向量，通过 softmax 归一化后作为两个分支的融合权重。融合后的特征经过 1×1 投影、全局池化和展平，形成 128 维的辅助表示向量。

该设计的优点在于显式区分了不同层级特征的功能，不再默认“越深越好”。对于年龄估计这类既依赖局部纹理又依赖整体结构的任务，这种双流融合能够更符合问题本身的表征需求。同时，由于投影通道控制在较小范围内，整体额外开销仍然较低。

## 3.4 Bottleneck Spatial Pyramid Pooling

在骨干网络深层，模型获得了 7×7 的高语义特征图。若直接使用全局平均池化，虽然实现简单，但可能损失部分空间上下文信息。为此，本文在深层特征后加入 Bottleneck SPP 模块，分别使用 5×5、9×9 和 13×13 的最大池化支路，再与原特征图进行拼接，最后通过 1×1 卷积压缩到 512 个通道。

其计算过程可表示为：

$$
p_1 = x,\quad
p_2 = \text{MaxPool}_{5 \times 5}(x),\quad
p_3 = \text{MaxPool}_{9 \times 9}(x),\quad
p_4 = \text{MaxPool}_{13 \times 13}(x)
$$

$$
f_{\text{spp}} = \text{Conv}_{1 \times 1}([p_1, p_2, p_3, p_4])
$$

SPP 的优势在于无需改变主干深度即可融合不同感受野的上下文。在年龄估计任务中，局部皱纹与全局结构的重要性会随年龄段变化而变化，多尺度上下文有助于模型对不同样本作出更稳健的判断。本文将 SPP 输出与双流融合特征拼接后输入预测头，使深层语义和中层细节能够同时参与最终决策。

## 3.5 标签分布学习与组合损失

本文采用标签分布学习来处理年龄模糊性问题。对于真实年龄 $y$，模型不再只构造 one-hot 标签，而是围绕该年龄生成一组离散高斯分布，并在输出空间内归一化。考虑到不同年龄段的不确定性程度并不完全相同，模型使用自适应 sigma 策略，使 sigma 随年龄大小变化。其中，配置文件中给出的参数为 $\sigma_{\min}=1.0$、$\sigma_{\max}=3.0$，对应输出空间最大年龄索引 80。

若年龄索引为 $j$，则目标分布可写为：

$$
P(j|x)=\frac{1}{Z}\exp\left(-\frac{(j-y)^2}{2\sigma^2}\right)
$$

其中，$\sigma$ 按年龄线性变化。当训练开启 sigma jitter 时，模型还会在一定范围内对 sigma 加入随机扰动，以增强分布监督的鲁棒性。

综合损失函数由四部分组成。第一部分是 KL 散度，作为主要分布匹配损失；第二部分是 L1 损失，直接约束期望年龄与真实年龄的偏差；第三部分是 ranking 损失，通过预测分布和目标分布的累积分布函数进行比较，增强年龄的序关系；第四部分是 Mean-Variance loss，用于同时约束预测均值与方差，鼓励输出分布既准确又相对集中。根据当前配置文件，整体损失的组合权重为：

$$
L = L_{KL} + 0.1L_{L1} + 0.5L_{rank} + 0.1L_{MV}
$$

其中，$L_{MV}$ 内部又包含均值误差项和方差惩罚项。该组合能够兼顾分布拟合、数值偏差、序关系和输出稳定性，是整体训练策略中的关键部分。

## 3.6 训练与推理策略

在训练阶段，本文采用 AdamW 优化器，学习率为 3×10^-4，权重衰减为 4×10^-4，批大小为 128，总轮数为 120。学习率调度使用余弦退火，并在 100 轮后保持在较低水平稳定收敛。为了提高训练初期的稳定性，模型前 10 轮冻结骨干网络的大部分参数，仅训练头部、SPP、融合层以及注入的 Coordinate Attention 模块；第 11 轮开始再进行全参数微调。

数据增强方面，本文在训练集上采用随机裁剪、水平翻转、仿射变换、高斯模糊、颜色抖动、MixUp 以及 Safe Random Erasing。其中 MixUp 的 alpha 为 0.5，触发概率为 0.5；Safe Random Erasing 的概率为 0.1，并通过启发式策略尽量避免完全覆盖关键面部区域。此外，模型启用 EMA 进行参数滑动平均，以提升验证与测试稳定性。

在验证与测试阶段，模型默认采用 6 次测试时增强，即 0.9、1.0、1.1 三个尺度分别结合原图和水平翻转图像求平均预测概率，再通过期望回归得到年龄结果。需要指出的是，本文实验章节给出的测试 MAE 均对应这种统一评估流程，因此在补充其他方法对比时，也应尽量保持相同或相近的评估口径。

结合训练脚本的实际行为，可以将本文采用的训练和评估流程总结为表 3-2。该表有助于在行文中明确区分不同阶段的目标与作用，避免将整套配置混写为单阶段训练。

| 阶段 | 轮次范围 | 关键行为 | 主要目的 |
|---|---|---|---|
| 阶段一 | 1-10 | 冻结骨干，仅训练头部、SPP、融合层和 CoordAtt | 稳定初始化新增模块 |
| 阶段二 | 11-100 | 全参数训练，余弦退火调度，持续 EMA，启用 MixUp 与 sigma jitter | 主体收敛与性能提升 |
| 阶段三 | 105-120 | 关闭 MixUp 与 sigma jitter，切换更干净的数据流 | 后期稳定收敛 |
| 验证/测试 | 全程 | 使用 0.9/1.0/1.1 三尺度与翻转的 6 次 TTA | 降低评估波动 |

表 3-2 基于训练脚本整理的三阶段训练与评估策略

# 4 实验设置、结果与分析

## 4.1 数据集与实验口径

本文实验使用预处理后的 AFAD 数据集。通过目录扫描可知，数据集总图像数为 163373 张，年龄目录从 15 岁延伸至 75 岁，但实际非空样本主要分布在 15 至 72 岁之间，且高龄样本明显稀缺。这一现象与年龄估计任务中常见的数据分布不均衡问题一致：青年与中青年样本数量较多，而高龄样本数量较少，容易导致模型在尾部年龄段的误差增大。

图 4-1 根据 AFAD 数据目录统计了样本年龄分布。可以看出，样本主要集中在青年与中青年区间，而高龄样本数量明显下降，这也是模型需要引入分布重加权与稳健损失设计的重要原因之一。

![图 4-1 AFAD 年龄分布图](output/doc/assets/afad_age_distribution.png)

为更清晰地展示数据构成，表 4-1 给出了 AFAD 数据集在本研究中的主要统计信息。可以看出，除总体样本量外，年龄覆盖区间与非空年龄段数量同样会影响后续训练稳定性和高龄预测性能。

| 统计项 | 数值 |
|---|---:|
| 图像总数 | 163373 |
| 年龄目录范围 | 15-75 |
| 实际非空年龄范围 | 15-72 |
| 非空年龄段数量 | 57 |
| 数据划分协议 | 72-8-20 |

表 4-1 AFAD 数据集统计信息

本文采用分层划分策略，保证不同年龄段样本在训练集、验证集和测试集中保持相对一致的比例。当前默认协议为 72-8-20，对应的划分文件为 `dataset_split_AFAD_72_8_20.json`。根据实际读取结果，训练集、验证集和测试集样本量如表 4-2 所示。

| 数据子集 | 样本数 | 比例 |
|---|---:|---:|
| 训练集 | 117603 | 72% |
| 验证集 | 13039 | 8% |
| 测试集 | 32731 | 20% |

表 4-2 AFAD 分层 72-8-20 数据划分结果

需要强调的是，本文统一采用上述分层 72-8-20 口径。这样做的原因在于代码配置、结果文件和相关说明均以该协议为主，若正文采用其他划分方式，会造成实验陈述和结果来源不一致。

## 4.2 实现细节与训练配置

模型的主要训练配置由 `src/config.py` 与 `src/train.py` 决定，关键超参数如表 4-3 所示。除表中参数外，训练过程中还启用了训练集年龄分布重加权、梯度裁剪和 AMP 混合精度，以提高训练效率与稳定性。

| 项目 | 配置值 |
|---|---|
| 输入尺寸 | 224×224 |
| 骨干网络 | MobileNetV3-Large |
| 类别空间 | 81（0-80） |
| 优化器 | AdamW |
| 初始学习率 | 3×10^-4 |
| 权重衰减 | 4×10^-4 |
| 批大小 | 128 |
| 训练轮数 | 120 |
| 冻结轮数 | 前 10 轮冻结骨干 |
| EMA 衰减 | 0.999 |
| MixUp | α=0.5，概率 0.5 |
| Random Erasing | 概率 0.1 |
| 自适应 sigma | 1.0 到 3.0 |
| 损失权重 | L1=0.1，Rank=0.5，MV=0.1 |

表 4-3 主要训练配置

从工程实现上看，训练过程分为三个阶段。第一阶段为冻结骨干的适应性训练，主要让新增模块先学会适应年龄估计任务；第二阶段为全参数优化，此时余弦退火调度开始发挥作用，EMA 也持续更新；第三阶段出现在训练后段，训练流程会关闭 MixUp 与 sigma jitter，并重建训练 DataLoader，使模型在更稳定的输入条件下收敛。验证与测试均使用相同的 6 次 TTA 流程，以减少单次前向预测的随机波动。

## 4.3 主要实验结果

根据结果文件，FADE-Net 在不同随机种子下的测试集 MAE 如表 4-4 所示。可以看到，模型在三个随机种子下的差异较小，表明其训练过程具有较好的稳定性。最佳单模型出现在 seed1337，对应测试集 MAE 为 3.0574；三次实验均值为 3.0857，标准差为 0.0204。

| Seed | 测试集 MAE |
|---|---:|
| 42 | 3.0951 |
| 1337 | 3.0574 |
| 2026 | 3.1047 |
| 平均值 ± 标准差 | 3.0857 ± 0.0204 |

表 4-4 多 seed 测试结果

本文将 seed1337 的结果作为最佳单模型性能，而将三 seed 结果作为补充稳定性分析。这样既能够突出模型的最佳表现，也能体现结果并非偶然。由于当前并未将集成结果作为统一比较口径，正文不直接引用 3.02 这一数值，以避免与单模型结果混淆。

为便于与典型年龄估计方法进行横向比较，本文参考公开文献中常见的 AFAD 指标，构造如表 4-5 所示的结果对比表。需要说明的是，不同文献在预处理、数据划分与评测协议上可能存在差异，因此该表更适合作为结果对照背景，而不是绝对严格的一一复现实验。

| 方法 | 骨干网络 | 参数量 | MAE |
|---|---|---:|---:|
| DEX | VGG-16 | 138M | 3.80 |
| OR-CNN | VGG-16 | 138M | 3.34 |
| RAN | ResNet-34 | 21.8M | 3.42 |
| CORAL | ResNet-34 | 21.8M | 3.48 |
| CDCNN | Multi-Task CNN | - | 3.11 |
| GRANET | ResNet-50 | 25.5M | 3.10 |
| FADE-Net（最佳单模型） | MobileNetV3-Large | 4.84M | 3.0574 |

表 4-5 AFAD 上的主要方法结果对比

从表 4-5 可以看出，FADE-Net 在参数规模远低于 VGG 和 ResNet 系列模型的情况下，依然取得了具有竞争力的 MAE。这表明针对年龄估计任务定制轻量结构，比单纯依赖更深更大的通用模型更具实际部署价值。

## 4.4 参数规模与训练过程分析

通过对模型参数进行实际计数，FADE-Net 的总参数量为 4841463，即 4.8415M；原始 torchvision `mobilenet_v3_large` 的参数量为 5483032，即 5.4830M。对比结果如表 4-6 所示。

| 模型 | 参数量 |
|---|---:|
| 原始 MobileNetV3-Large | 5.4830M |
| FADE-Net | 4.8415M |
| 参数减少量 | 0.6415M |

表 4-6 参数规模对比

这一下降主要来自于任务特定头部对原始分类头的替换。虽然 FADE-Net 在中间层增加了双流融合和 SPP 等模块，但移除了面向 1000 类 ImageNet 分类的大型全连接头，并重新构造为更适合年龄分布预测的轻量头部，因此总参数量反而低于原始骨干。

为观察训练过程是否稳定，本文引用了 seed1337 对应的训练曲线。图 4-2 为训练损失曲线，图 4-3 为验证 MAE 曲线。从变化趋势看，模型在前期收敛较快，中后期进入稳定下降阶段，说明冻结骨干、EMA 和分阶段增强策略具有实际作用。

![图 4-2 训练损失曲线](plots/seed_1337/1_loss_curve.png)

![图 4-3 验证集 MAE 曲线](plots/seed_1337/2_mae_curve.png)

进一步地，图 4-4 和图 4-5 展示了学习率调度与 batch 稳定性曲线。前者能够说明余弦退火在 100 轮前后的变化趋势，后者则有助于观察训练过程中批次损失是否存在剧烈震荡。二者共同反映了训练策略在工程上具有较好的平稳性。

![图 4-4 学习率调度曲线](plots/seed_1337/3_lr_schedule.png)

![图 4-5 Batch 稳定性曲线](plots/seed_1337/5_batch_stability.png)

图 4-6 给出了训练与验证之间的泛化差距曲线，图 4-7 则展示了时间效率统计结果。前者可用于观察模型是否在中后期出现明显过拟合，后者则能够从另一个角度说明轻量化设计在工程应用中的潜在价值。结合前述参数规模分析，可以认为 FADE-Net 在精度与效率之间取得了较为均衡的结果。

![图 4-6 泛化差距曲线](plots/seed_1337/4_generalization_gap.png)

![图 4-7 时间效率统计图](plots/seed_1337/6_time_efficiency.png)

## 4.5 结果讨论与不足分析

综合上述结果，可以从三个角度理解 FADE-Net 的性能表现。第一，从结构角度看，深层注意力注入和中层双流融合为轻量骨干补足了年龄估计任务所需的空间敏感性与多尺度表征；第二，从监督角度看，标签分布学习与排序损失使模型不再只关注单点年龄，而是通过概率分布描述年龄的不确定性；第三，从训练角度看，冻结策略、EMA 和多尺度测试增强共同提高了验证与测试阶段的稳定性。

当然，当前结果也存在一定局限。其一，数据分布不均衡问题仍然突出，高龄样本数量过少可能导致模型在尾部年龄段误差增大；其二，尽管本文已经从结构、训练与统计曲线三个角度进行了分析，但尚未进一步补充基于同一评测协议的完整消融结果与注意力可视化结果；其三，表 4-5 中的横向对比受到公开文献划分协议差异的影响，因此其意义更偏向于结果参照，而不是严格复现实验。

# 5 总结与展望

## 5.1 全文总结

本文围绕“基于注意力机制的轻量化人脸年龄估计”这一主题，结合模型实现与实验结果，对 FADE-Net 模型进行了系统梳理。全文首先分析了人脸年龄估计任务在应用场景、研究价值和技术难点上的背景，指出现有高精度方法往往参数量较大、部署成本较高，而轻量骨干若缺乏针对性的结构增强，则容易出现年龄特征表达不足的问题。

在方法部分，本文重点介绍了 FADE-Net 的几项核心设计：一是在 MobileNetV3-Large 深层使用 Pyramid Attention Injection，将最后 4 个 SE 模块替换为 Coordinate Attention；二是从 Block 6 和 Block 12 提取纹理与语义特征，并通过可学习权重进行双流融合；三是在深层语义分支中加入 Bottleneck SPP，以低成本引入多尺度上下文；四是基于标签分布学习、L1、CDF ranking 和 Mean-Variance loss 构建组合损失，从而增强模型对年龄模糊性和序关系的建模能力。

在实验部分，本文采用 AFAD 分层 72-8-20 协议，对结果文件进行了统一整理与分析。结果显示，模型在 seed1337 下取得 3.0574 的最佳单模型 MAE，三次实验均值为 3.0857 ± 0.0204，参数量仅为 4.8415M。这说明所提出的轻量化结构具备较好的精度效率平衡，为移动端与边缘端人脸年龄估计提供了一种可行方案。

总体来看，现有结果较好地支撑了本文的研究思路。实验分析表明，面向年龄估计任务进行轻量化与任务特定增强设计，能够在不显著增加结构复杂度的前提下取得稳定有效的性能收益。

## 5.2 后续展望

未来工作可以从以下几个方面继续展开。首先，在实验层面，可进一步补充完整消融实验，对 HA、DLDL、MSFF、SPP 和 MV 等模块进行独立验证，并增加均值与标准差统计，使论证更为完整。其次，在可视化层面，可结合 Grad-CAM 或特征响应图进一步展示模型对额头、眼角、嘴角等关键区域的关注模式。再次，在泛化层面，可尝试将模型迁移到 Morph、CACD、UTKFace 等其他数据集，以验证其跨数据集鲁棒性。最后，在部署层面，仍可进一步考虑量化、剪枝与蒸馏等模型压缩技术，以继续降低推理延迟和存储成本。

# 参考文献

[1] Rothe R, Timofte R, Van Gool L. DEX: Deep EXpectation of apparent age from a single image[C]//ICCV Workshops. 2015.

[2] Niu Z, Zhou M, Wang L, Gao X, Hua G. Ordinal regression with multiple output CNN for age estimation[C]//CVPR. 2016.

[3] Gao B B, Xing C, Xie C W, Wu J, Geng X. Deep label distribution learning with label ambiguity[J]. IEEE Transactions on Image Processing, 2017.

[4] Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

[5] Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[6] He K, Zhang X, Ren S, Sun J. Spatial pyramid pooling in deep convolutional networks for visual recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015.

[7] Garain A, Ray R, Singh P K, et al. GRA_Net: A deep learning model for classification of age and gender from facial images[J]. IEEE Access, 2021.

[8] Cao W, Mirjalili V, Raschka S. Rank consistent ordinal regression for neural networks with application to age estimation[J]. Pattern Recognition Letters, 2020.

[9] Wang F, Jiang M, Qian C, et al. Residual attention network for image classification[C]//CVPR. 2017.

[10] Mehta S, Rastegari M. MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer[C]//ICLR. 2022.

[11] Xi J, Xu Z, Yan Z, Liu W, Liu Y. Portrait age recognition method based on improved ResNet and deformable convolution[J]. Electronic Research Archive, 2023.

[12] Bekhouche S E, Benlamoudi A, Dornaika F, Telli H, Bounab Y. Facial age estimation using multi-stage deep neural networks[J]. Electronics, 2024.

[13] 王一帆, 孙辉, 张静, 等. 基于深度学习的人脸年龄估计研究综述[J]. 计算机工程与应用, 2023.

# 致谢

本文的研究与撰写过程中得到了多方面的支持与帮助。首先，感谢指导教师在研究思路、论文结构与方法设计方面给予的指导。其次，感谢在模型实现、训练调试、数据处理和材料整理过程中提供帮助的相关人员。最后，感谢公开数据集、开源框架与相关研究工作的贡献者，为本文的实现与分析提供了重要基础。在此谨致谢忱。
