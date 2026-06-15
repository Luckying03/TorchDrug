# 基于 MindSpore 的 TorchDrug 论文简化复现

本项目用于研究生课程大作业。选题论文为 **TorchDrug: A Powerful and Flexible Machine Learning Platform for Drug Discovery**。原论文提出了面向药物发现的机器学习平台 TorchDrug，本项目不复现完整平台，而是选取其中最典型的分子性质预测任务做简化复现。

本项目遵守课程要求：

- 不直接调用 TorchDrug 或 PyTorch 训练模型。
- 使用 RDKit 处理 SMILES 并构造分子图。
- 使用 MindSpore 实现 GIN/GAT、loss、训练循环和评估。
- 在 BACE 和 HIV 数据集上完成分子二分类活性预测。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── check_env.py
├── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── dataset.py        # 下载 CSV、RDKit 构图、scaffold/random split、edge-list batch
│   ├── features.py       # TorchDrug default atom/bond 特征与 SMILES -> graph
│   ├── metrics.py        # AUROC / AUPRC
│   ├── models.py         # MindSpore GIN / GAT
│   └── trainer.py        # MindSpore loss、optimizer、训练与评估循环
├── notebooks/
│   └── TorchDrug_MindSpore_Reproduction.ipynb
├── results/
│   └── experiment_results.csv
└── docs/
    ├── report_outline.md
    └── ppt_outline.md
```

运行实验后会自动生成 `data/` 缓存目录，保存下载的 CSV 和处理后的分子图缓存。

## 环境安装

建议使用 Python 3.9 或 3.10。MindSpore 不同硬件后端安装方式不同，优先参考 MindSpore 官方安装页面选择 CPU / GPU / Ascend 对应命令。

CPU 环境示例：

```bash
conda create -n ms-gnn python=3.9 -y
conda activate ms-gnn
pip install mindspore
conda install -c conda-forge rdkit -y
pip install numpy scikit-learn pandas matplotlib notebook ipykernel
```

如果你的系统无法直接安装 MindSpore，Windows 用户可以考虑使用 WSL2 / Linux 环境完成实验。

检查环境：

```bash
python check_env.py
```

## Notebook 运行

Notebook 文件：

```text
notebooks/TorchDrug_MindSpore_Reproduction.ipynb
```

Notebook 中每个代码块前都有 Markdown 说明，适合课堂展示和报告截图。进入项目根目录后启动：

```bash
jupyter notebook
```

然后打开 `notebooks/TorchDrug_MindSpore_Reproduction.ipynb`。

## 命令行运行实验

冒烟测试：

```bash
python run_experiment.py --dataset bace --model gin --epoch 1 --batch_size 32 --seed 0 --device_target GPU
```

四组核心实验：

```bash
python run_experiment.py --dataset bace --model gin --epoch 100 --batch_size 256 --seed 0 --device_target GPU --selection best_valid_auroc
python run_experiment.py --dataset bace --model gat --epoch 100 --batch_size 256 --seed 0 --device_target GPU --selection best_valid_auroc
python run_experiment.py --dataset hiv --model gin --epoch 100 --batch_size 256 --seed 0 --split random --device_target GPU --selection best_valid_auroc
python run_experiment.py --dataset hiv --model gat --epoch 100 --batch_size 256 --seed 0 --split random --device_target GPU --selection best_valid_auroc
```

一次运行 BACE / HIV 与 GIN / GAT 的 2×2 组合：

```bash
python run_experiment.py --dataset all --model all --epoch 100 --batch_size 256 --seed 0 --device_target GPU --selection best_valid_auroc
```

多个随机种子：

```bash
python run_experiment.py --dataset all --model all --epoch 100 --batch_size 256 --seed 1 --device_target GPU --selection best_valid_auroc
python run_experiment.py --dataset all --model all --epoch 100 --batch_size 256 --seed 2 --device_target GPU --selection best_valid_auroc
```

默认使用 scaffold split。若需要随机划分对照：

```bash
python run_experiment.py --dataset bace --model gin --epoch 100 --batch_size 256 --seed 0 --split random --device_target GPU --selection best_valid_auroc
```

## 对比实验

默认 `--variant torchdrug_like` 会尽量贴近 TorchDrug 的默认实现：

- 图表示：参考 TorchDrug 的 sparse edge-list，而不是 dense 邻接矩阵。
- 节点特征：复刻 TorchDrug `features.atom.default`。
- 边特征：复刻 TorchDrug `features.bond.default`。
- 模型结构：edge feature 输入、BatchNorm、shortcut、concat hidden、sum readout。
- 默认隐藏维度：256。

若要和早期教学简化版对比，可以运行 `--variant simple`：

```bash
python run_experiment.py --dataset bace --model gin --variant simple --epoch 100 --batch_size 256 --seed 0 --device_target GPU --selection best_valid_auroc
python run_experiment.py --dataset bace --model gin --variant torchdrug_like --epoch 100 --batch_size 256 --seed 0 --device_target GPU --selection best_valid_auroc
```

## 命令行参数

核心参数：

```text
--dataset        bace / hiv / all
--model          gin / gat / all
--epoch          训练轮数
--batch_size     batch size
--seed           随机种子
```

其他参数：

```text
--split          scaffold / random，默认 scaffold
--selection      best_valid_auroc / final，默认 best_valid_auroc
--device_target  CPU / GPU / Ascend，默认 CPU
--lr             学习率，默认 1e-3
--hidden_dim     隐藏层维度，默认 256
--num_layer      GNN 层数，默认 4
--num_head       GAT 注意力头数，默认 4
--dropout        dropout，默认 0.1
--variant        torchdrug_like / simple，默认 torchdrug_like
--readout        sum / mean，torchdrug_like 默认 sum
--num_mlp_layer  图级预测头 MLP 层数，torchdrug_like 默认 1
```

## 实验设计

数据集：

- BACE：BACE-1 抑制剂二分类任务，标签字段为 `Class`。
- HIV：HIV replication inhibition 二分类任务，标签字段为 `HIV_active`。

分子图构造：

- 使用 RDKit 解析 SMILES。
- 原子作为节点，化学键作为无向边。
- 默认 `torchdrug_like` 特征复刻 TorchDrug `atom_default`：原子符号、手性、度数、形式电荷、氢原子数、自由基电子数、杂化、芳香性和环信息。
- 默认 `torchdrug_like` 边特征复刻 TorchDrug `bond_default`：键类型、键方向、立体构型和共轭信息。
- batch 内拼接所有分子的节点与边，形成 `node_feature`、`edge_list`、`edge_feature`、`node2graph`。这与 TorchDrug 的 sparse edge-list 思路一致，避免 dense 邻接矩阵的 `O(N^2)` 显存开销。

模型：

- GIN：邻居求和聚合，edge feature 线性映射后加入消息，MLP 更新节点表示，默认 sum readout。
- GAT：多头注意力，edge feature 加入 attention key，基于邻接矩阵做 masked softmax，默认 sum readout。
- `torchdrug_like` 变体启用 BatchNorm、shortcut、concat hidden，以贴近 TorchDrug GIN/GAT 参数。

训练：

- MindSpore `BCEWithLogitsLoss`。
- MindSpore `Adam` 优化器。
- 指标为 AUROC 和 AUPRC。

## 结果表

结果会追加写入：

```text
results/experiment_results.csv
```

字段包括：

```text
timestamp,framework,dataset,model,variant,graph_format,feature_set,seed,split,selection,selected_epoch,
epoch,batch_size,hidden_dim,num_layer,num_head,readout,num_mlp_layer,node_feature_dim,edge_feature_dim,
valid_auroc,valid_auprc,test_auroc,test_auprc
```

课程要求的 `dataset`、`model`、`seed`、valid/test AUROC、valid/test AUPRC 都已经包含。

## 报告和 PPT

报告大纲：

```text
docs/report_outline.md
```

PPT 大纲：

```text
docs/ppt_outline.md
```

建议报告中至少展示：

- 原论文核心思想。
- 本项目与原 TorchDrug 平台的关系和简化点。
- RDKit 分子图构造方法。
- MindSpore GIN/GAT 实现。
- BACE/HIV 上的结果表。
- AUROC/AUPRC 指标分析。

## 参考链接

- TorchDrug 论文：https://arxiv.org/abs/2202.08320
- TorchDrug 项目：https://github.com/DeepGraphLearning/torchdrug
- MindSpore 安装：https://www.mindspore.cn/install
- RDKit：https://www.rdkit.org/

## 建议的提交内容

可以提交整个项目目录，但不要提交运行生成的大缓存：

```text
README.md
requirements.txt
check_env.py
run_experiment.py
src/
notebooks/
results/experiment_results.csv
docs/
.gitignore
```
