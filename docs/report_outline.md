# 研究报告大纲

## 题目

基于 MindSpore 的 TorchDrug 论文简化复现：面向 BACE 与 HIV 的分子二分类活性预测

## 1. 研究背景

- 药物虚拟筛选希望在大量候选分子中优先发现潜在活性分子。
- 分子天然可以表示为图：原子是节点，化学键是边。
- 图神经网络可以从分子图结构中学习分子表示，用于活性预测、毒性预测等任务。

## 2. 论文概述

- 论文：TorchDrug: A Powerful and Flexible Machine Learning Platform for Drug Discovery。
- 原论文贡献是提供药物发现任务的数据、模型、任务封装与训练评估平台。
- 本作业不复现完整平台，而是选取其中分子性质预测任务进行简化复现。
- 复现重点：使用 GIN/GAT 图神经网络进行 BACE 与 HIV 二分类预测。

## 3. 数据集

- BACE：预测分子是否为 BACE-1 抑制剂，标签字段为 `Class`。
- HIV：预测分子是否具有抑制 HIV replication 的活性，标签字段为 `HIV_active`。
- 数据来源：公开 DeepChem / MoleculeNet CSV。
- 数据处理：RDKit 解析 SMILES，构造分子图节点特征和邻接矩阵。

## 4. 方法

### 4.1 分子图表示

- 为尽量贴近 TorchDrug，本项目默认复刻 `features.atom.default` 与 `features.bond.default`。
- 节点特征包括原子符号、手性、度数、形式电荷、氢原子数、自由基电子数、杂化、芳香性和环信息，共 69 维。
- 边特征包括键类型、键方向、立体构型和共轭信息，共 19 维。
- 邻接矩阵由 RDKit 化学键生成，使用无向图表示分子结构。

### 4.2 GIN 模型

- 每层计算邻居节点表示求和。
- 边特征经过线性映射后加入消息聚合，贴近 TorchDrug 的 `edge_input_dim` 机制。
- 使用 MLP 更新节点表示。
- 使用 BatchNorm、shortcut、concat hidden 和 sum readout，贴近 TorchDrug GIN 默认可选结构。

### 4.3 GAT 模型

- 使用多头注意力计算相邻节点的重要性。
- 边特征加入 attention key，贴近 TorchDrug GAT 的 edge-aware attention。
- 在邻接矩阵范围内做 masked softmax。
- 聚合邻居表示并得到图级表示。

### 4.4 训练目标

- 使用 MindSpore 实现模型前向、BCEWithLogitsLoss、Adam 优化器和训练循环。
- 不调用 TorchDrug 或 PyTorch 训练代码。

## 5. 实验设置

- 框架：MindSpore。
- 数据划分：默认 scaffold split，比例为 8:1:1。
- 评价指标：AUROC、AUPRC。
- 对比模型：GIN、GAT。
- 对比变体：`torchdrug_like` 与 `simple`。前者尽量对齐 TorchDrug，后者为早期简化基线。
- 随机种子：建议至少使用 0、1、2 三组。
- 记录字段：dataset、model、seed、valid/test AUROC、valid/test AUPRC。

## 6. 实验结果

将 `results/experiment_results.csv` 整理为表格：

| Dataset | Model | Seed | Valid AUROC | Valid AUPRC | Test AUROC | Test AUPRC |
| --- | --- | --- | --- | --- | --- | --- |
| BACE | GIN | 0 |  |  |  |  |
| BACE | GAT | 0 |  |  |  |  |
| HIV | GIN | 0 |  |  |  |  |
| HIV | GAT | 0 |  |  |  |  |

## 7. 结果分析

- 比较 GIN 与 GAT 在 BACE 上的表现差异。
- 比较 GIN 与 GAT 在 HIV 上的表现差异。
- 分析 AUROC 与 AUPRC 的含义，尤其说明 AUPRC 对类别不均衡任务更敏感。
- 讨论 scaffold split 对模型泛化评估的意义。

## 8. 局限性

- 只复现了 TorchDrug 论文中的分子性质预测思路，没有复现完整平台。
- 使用了简化节点特征和稠密邻接矩阵，效率不如专门图学习框架。
- 未使用预训练模型和大规模超参数搜索。
- 训练结果受随机种子、硬件和依赖版本影响。

## 9. 结论

- 本项目基于 MindSpore 实现了 GIN/GAT 分子图分类模型。
- 在 BACE 和 HIV 上完成了分子二分类活性预测流程。
- 复现实验证明，TorchDrug 论文中“统一数据处理、图模型、任务封装和指标评估”的核心思路可以迁移到 MindSpore 实现。
