# 基于 MindSpore 的 TorchDrug 复现

选题论文：**TorchDrug: A Powerful and Flexible Machine Learning Platform for Drug Discovery**。

本项目复现其中的分子性质预测任务：用 RDKit 从 SMILES 构造分子图，用 MindSpore 实现 GIN/GAT，在 BACE 和 HIV 数据集上做二分类活性预测。代码不直接调用 PyTorch 或 TorchDrug 的训练流程。

## 目录

```text
.
├── README.md
├── requirements.txt
├── check_env.py
├── run.ipynb
├── src/
│   ├── dataset.py      # 数据下载、SMILES 解析、数据划分、batch 拼接
│   ├── features.py     # 原子和化学键特征
│   ├── metrics.py      # AUROC / AUPRC
│   ├── models.py       # MindSpore 版 GIN / GAT
│   └── trainer.py      # loss、优化器、训练和评估
├── results/          # 运行后保存实验结果
└── latex/
    ├── main.tex
    └── main.pdf
```

`data/` 会在运行时自动生成，用来缓存下载的数据和处理后的分子图。

## 环境

我在华为云 MindSpore CUDA 镜像中运行，环境名为 `MindSpore`。

```bash
conda activate MindSpore
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python check_env.py
```


## 运行


打开 `run.ipynb`，选择 `MindSpore` 内核，然后按顺序运行全部代码块。

Notebook 中主要参数：

```python
DEVICE_TARGET = "GPU"
FULL_EPOCH = 100
SEED = 0
```

## 实验设置

四组实验都在 `run.ipynb` 中完成：

```text
BACE + GIN   scaffold split
BACE + GAT   scaffold split
HIV  + GIN   random split
HIV  + GAT   random split
```

默认设置为 hidden dim 256、GNN 层数 4、学习率 1e-3，结果选择验证集 AUROC 最优的轮次。

运行结束后结果会保存到：

```text
results/notebook_experiment_results.csv
```

结果字段包括数据集、模型、seed、split、selected epoch、valid/test AUROC 和 valid/test AUPRC。
