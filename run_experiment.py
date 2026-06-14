"""Run TorchDrug molecular property prediction experiments.

This script intentionally lives outside the TorchDrug source tree. It uses the
local DeepGraphLearning/torchdrug checkout if it exists, but never modifies it.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "README.md").exists() else SCRIPT_DIR.parent
TORCHDRUG_CANDIDATES = [
    PROJECT_ROOT / "torchdrug",
    PROJECT_ROOT.parent / "torchdrug",
]
for local_torchdrug in TORCHDRUG_CANDIDATES:
    if (local_torchdrug / "torchdrug" / "__init__.py").exists():
        sys.path.insert(0, str(local_torchdrug))
        break

np = None
torch = None
core = None
data = None
datasets = None
models = None
tasks = None
DATASETS = {}


def import_training_dependencies() -> None:
    global np, torch, core, data, datasets, models, tasks, DATASETS

    try:
        import numpy as _np
        import torch as _torch
        from torchdrug import core as _core
        from torchdrug import data as _data
        from torchdrug import datasets as _datasets
        from torchdrug import models as _models
        from torchdrug import tasks as _tasks
    except ModuleNotFoundError as exc:
        message = (
            "TorchDrug or one of its dependencies cannot be imported.\n"
            f"Missing module: {exc.name}\n"
            "Run `python experiments/check_env.py` for a fuller diagnosis, then "
            "install the recommended conda environment from README.md."
        )
        raise SystemExit(message) from exc

    np = _np
    torch = _torch
    core = _core
    data = _data
    datasets = _datasets
    models = _models
    tasks = _tasks
    DATASETS = {
        "bace": datasets.BACE,
        "hiv": datasets.HIV,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GIN/GAT on TorchDrug BACE or HIV with PropertyPrediction."
    )
    parser.add_argument("--dataset", choices=["bace", "hiv", "all"], required=True)
    parser.add_argument("--model", choices=["gin", "gat", "all"], required=True)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--result_csv", type=Path, default=PROJECT_ROOT / "results" / "results.csv")
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--gat_num_head", type=int, default=4)
    parser.add_argument("--lazy", action="store_true", help="Build molecules lazily to reduce memory use.")
    parser.add_argument("--quiet", action="store_true", help="Reduce TorchDrug dataset logging.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_gpus(device: str) -> List[int] | None:
    if device == "cpu":
        return None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")
        return [0]
    if torch.cuda.is_available():
        return [0]
    return None


def load_dataset(name: str, data_dir: Path, lazy: bool, verbose: int):
    dataset_dir = data_dir / name
    dataset_cls = DATASETS[name]
    return dataset_cls(str(dataset_dir), lazy=lazy, verbose=verbose)


def make_split(dataset, split: str, seed: int):
    train_len = int(0.8 * len(dataset))
    valid_len = int(0.1 * len(dataset))
    lengths = [train_len, valid_len, len(dataset) - train_len - valid_len]

    if split == "scaffold":
        torch.manual_seed(seed)
        return data.scaffold_split(dataset, lengths)

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, lengths, generator=generator)


def build_model(model_name: str, dataset, hidden_dim: int, num_layer: int, gat_num_head: int):
    hidden_dims = [hidden_dim] * num_layer
    kwargs = dict(
        input_dim=dataset.node_feature_dim,
        hidden_dims=hidden_dims,
        edge_input_dim=dataset.edge_feature_dim,
        short_cut=True,
        batch_norm=True,
        concat_hidden=True,
        readout="mean",
    )
    if model_name == "gin":
        return models.GIN(num_mlp_layer=2, **kwargs)
    if hidden_dim % gat_num_head != 0:
        raise ValueError("`--hidden_dim` must be divisible by `--gat_num_head` for GAT.")
    return models.GAT(num_head=gat_num_head, **kwargs)


def metric_value(metrics: Dict[str, torch.Tensor], metric_name: str) -> float:
    for key, value in metrics.items():
        if key.startswith(metric_name):
            value = value.detach().cpu().float()
            return float(value.mean().item())
    available = ", ".join(metrics.keys())
    raise KeyError(f"Cannot find metric `{metric_name}` in: {available}")


def append_result(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "dataset",
        "model",
        "seed",
        "split",
        "epoch",
        "batch_size",
        "valid_auroc",
        "valid_auprc",
        "test_auroc",
        "test_auprc",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one(args: argparse.Namespace, dataset_name: str, model_name: str) -> Dict[str, object]:
    print(f"\n=== dataset={dataset_name} model={model_name} seed={args.seed} ===")
    set_seed(args.seed)

    dataset = load_dataset(
        dataset_name,
        args.data_dir,
        lazy=args.lazy,
        verbose=0 if args.quiet else 1,
    )
    train_set, valid_set, test_set = make_split(dataset, args.split, args.seed)
    print(
        "Dataset loaded: "
        f"{len(dataset)} samples, tasks={dataset.tasks}, "
        f"split=({len(train_set)}, {len(valid_set)}, {len(test_set)})"
    )

    model = build_model(
        model_name,
        dataset,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        gat_num_head=args.gat_num_head,
    )
    task = tasks.PropertyPrediction(
        model,
        task=dataset.tasks,
        criterion="bce",
        metric=("auroc", "auprc"),
        num_mlp_layer=2,
    )
    optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)
    solver = core.Engine(
        task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        gpus=select_gpus(args.device),
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        log_interval=100,
    )

    solver.train(num_epoch=args.epoch)
    valid_metrics = solver.evaluate("valid")
    test_metrics = solver.evaluate("test")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": model_name,
        "seed": args.seed,
        "split": args.split,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "valid_auroc": metric_value(valid_metrics, "auroc"),
        "valid_auprc": metric_value(valid_metrics, "auprc"),
        "test_auroc": metric_value(test_metrics, "auroc"),
        "test_auprc": metric_value(test_metrics, "auprc"),
    }
    append_result(args.result_csv, row)
    print(
        "Saved result: "
        f"valid AUROC={row['valid_auroc']:.4f}, valid AUPRC={row['valid_auprc']:.4f}, "
        f"test AUROC={row['test_auroc']:.4f}, test AUPRC={row['test_auprc']:.4f}"
    )
    return row


def expand_choice(choice: str, values: Iterable[str]) -> List[str]:
    return list(values) if choice == "all" else [choice]


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    import_training_dependencies()

    dataset_names = expand_choice(args.dataset, DATASETS.keys())
    model_names = expand_choice(args.model, ["gin", "gat"])
    for dataset_name in dataset_names:
        for model_name in model_names:
            run_one(args, dataset_name, model_name)

    print(f"\nAll requested experiments finished. CSV: {args.result_csv}")


if __name__ == "__main__":
    main()
