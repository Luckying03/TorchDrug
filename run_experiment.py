"""Run MindSpore GIN / GAT experiments on BACE and HIV."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindSpore molecular activity prediction.")
    parser.add_argument("--dataset", choices=["bace", "hiv", "all"], required=True)
    parser.add_argument("--model", choices=["gin", "gat", "all"], required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--selection", choices=["best_valid_auroc", "final"], default="best_valid_auroc")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--result_csv", type=Path, default=PROJECT_ROOT / "results" / "experiment_results.csv")
    parser.add_argument("--device_target", choices=["CPU", "GPU", "Ascend"], default="CPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--variant", choices=["torchdrug_like", "simple"], default="torchdrug_like")
    parser.add_argument("--readout", choices=["sum", "mean"], default="sum")
    parser.add_argument("--num_mlp_layer", type=int, default=1)
    parser.add_argument("--no_cache", action="store_true", help="Disable processed graph cache.")
    return parser.parse_args()


def import_dependencies():
    try:
        import mindspore as ms
        import numpy as np
        from src.dataset import load_dataset, split_dataset
        from src.models import build_model
        from src.trainer import TrainConfig, fit
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: "
            f"{exc.name}\nRun `python check_env.py` and install packages from README.md."
        ) from exc
    return ms, np, load_dataset, split_dataset, build_model, TrainConfig, fit


def set_seed(ms, np, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def append_result(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "framework",
        "dataset",
        "model",
        "variant",
        "graph_format",
        "feature_set",
        "seed",
        "split",
        "selection",
        "selected_epoch",
        "epoch",
        "batch_size",
        "hidden_dim",
        "num_layer",
        "num_head",
        "readout",
        "num_mlp_layer",
        "node_feature_dim",
        "edge_feature_dim",
        "valid_auroc",
        "valid_auprc",
        "test_auroc",
        "test_auprc",
    ]
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8") as fin:
            old_header = fin.readline().strip()
        new_header = ",".join(fieldnames)
        if old_header != new_header:
            backup = csv_path.with_name(f"{csv_path.stem}_legacy{csv_path.suffix}")
            csv_path.replace(backup)
            print(f"Existing result schema differs. Moved old result file to: {backup}")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def expand_choice(choice: str, values: Iterable[str]) -> List[str]:
    return list(values) if choice == "all" else [choice]


def run_one(args: argparse.Namespace, dataset_name: str, model_name: str) -> Dict[str, object]:
    ms, np, load_dataset, split_dataset, build_model, TrainConfig, fit = import_dependencies()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    set_seed(ms, np, args.seed)

    print(f"\n=== MindSpore | dataset={dataset_name} model={model_name} seed={args.seed} ===")
    feature_set = "simple" if args.variant == "simple" else "torchdrug_default"
    effective_readout = "mean" if args.variant == "simple" else args.readout
    effective_num_mlp_layer = 2 if args.variant == "simple" else args.num_mlp_layer
    dataset = load_dataset(dataset_name, args.data_dir, use_cache=not args.no_cache, feature_set=feature_set)
    train_set, valid_set, test_set = split_dataset(dataset, args.split, args.seed)
    print(
        f"split sizes: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)} | "
        f"node feature dim={dataset.node_feature_dim}, edge feature dim={dataset.edge_feature_dim}"
    )
    print(
        f"hyperparameters: hidden_dim={args.hidden_dim}, num_layer={args.num_layer}, "
        f"batch_size={args.batch_size}, epoch={args.epoch}, selection={args.selection}"
    )

    model = build_model(
        model_name,
        input_dim=dataset.node_feature_dim,
        edge_input_dim=dataset.edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        num_head=args.num_head,
        dropout=args.dropout,
        variant=args.variant,
        readout=args.readout,
        num_mlp_layer=args.num_mlp_layer,
    )
    config = TrainConfig(
        epoch=args.epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        selection=args.selection,
    )
    valid_metrics, test_metrics, selected_epoch = fit(model, train_set, valid_set, test_set, config)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": "MindSpore",
        "dataset": dataset_name,
        "model": model_name,
        "variant": args.variant,
        "graph_format": "edge_list",
        "feature_set": feature_set,
        "seed": args.seed,
        "split": args.split,
        "selection": args.selection,
        "selected_epoch": selected_epoch,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layer": args.num_layer,
        "num_head": args.num_head,
        "readout": effective_readout,
        "num_mlp_layer": effective_num_mlp_layer,
        "node_feature_dim": dataset.node_feature_dim,
        "edge_feature_dim": dataset.edge_feature_dim,
        "valid_auroc": valid_metrics["auroc"],
        "valid_auprc": valid_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
    }
    append_result(args.result_csv, row)
    print(
        "saved result | "
        f"selection={args.selection} epoch={selected_epoch} | "
        f"valid AUROC={row['valid_auroc']:.4f}, valid AUPRC={row['valid_auprc']:.4f}, "
        f"test AUROC={row['test_auroc']:.4f}, test AUPRC={row['test_auprc']:.4f}"
    )
    return row


def main() -> None:
    args = parse_args()
    for dataset_name in expand_choice(args.dataset, ["bace", "hiv"]):
        for model_name in expand_choice(args.model, ["gin", "gat"]):
            run_one(args, dataset_name, model_name)
    print(f"\nFinished. Results: {args.result_csv}")


if __name__ == "__main__":
    main()
