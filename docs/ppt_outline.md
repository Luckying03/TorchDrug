# PPT 大纲

## 第 1 页：标题页

- 基于 MindSpore 的 TorchDrug 论文简化复现
- BACE / HIV 分子二分类活性预测
- 姓名、课程、日期

## 第 2 页：研究背景

- 药物发现成本高、周期长。
- 虚拟筛选可提前筛选候选分子。
- 分子图表示适合图神经网络建模。

## 第 3 页：原论文简介

- TorchDrug 是面向药物发现的机器学习平台。
- 支持数据集、图神经网络模型、任务封装和评估指标。
- 本项目复现其中分子性质预测任务的核心流程。

## 第 4 页：复现目标

- 不直接调用 TorchDrug 训练代码。
- 使用 RDKit 从 SMILES 构造分子图。
- 使用 MindSpore 实现 GIN/GAT、loss、训练循环和评估。
- 在 BACE 和 HIV 数据集上预测分子活性。

## 第 5 页：数据集

- BACE：BACE-1 抑制剂二分类。
- HIV：HIV replication inhibition 二分类。
- 输入：SMILES。
- 输出：二分类标签。

## 第 6 页：数据处理流程

- 下载公开 CSV。
- RDKit 解析 SMILES。
- 复刻 TorchDrug default atom feature，节点特征 69 维。
- 复刻 TorchDrug default bond feature，边特征 19 维。
- 构造邻接矩阵和边特征张量。
- 进行 scaffold split。

## 第 7 页：GIN 模型

- 邻居节点求和聚合。
- 边特征线性映射后加入消息。
- MLP 更新节点表示。
- BatchNorm、shortcut、concat hidden、sum readout。
- 适合捕捉分子局部结构模式。

## 第 8 页：GAT 模型

- 多头注意力计算邻居权重。
- 边特征加入 attention key。
- masked softmax 限制在分子图边上。
- 加权聚合节点表示。

## 第 9 页：实验设置

- 框架：MindSpore。
- 复现变体：torchdrug_like。
- 对照变体：simple。
- Loss：BCEWithLogitsLoss。
- Optimizer：Adam。
- Metrics：AUROC、AUPRC。
- Split：scaffold split 8:1:1。

## 第 10 页：实验结果

展示结果表：

| Dataset | Model | Test AUROC | Test AUPRC |
| --- | --- | --- | --- |
| BACE | GIN |  |  |
| BACE | GAT |  |  |
| HIV | GIN |  |  |
| HIV | GAT |  |  |

## 第 11 页：结果分析

- 比较 GIN 与 GAT。
- 分析 BACE 与 HIV 的差异。
- 说明 AUPRC 在类别不均衡任务中的意义。
- 讨论 scaffold split 下测试集表现。

## 第 12 页：总结与展望

- 完成了 MindSpore 版本的简化复现。
- 验证了分子图神经网络在活性预测中的可行性。
- 后续可加入边特征、预训练模型、更多数据集和更充分调参。
