"""Dataset download, SMILES parsing, splitting and batching utilities."""

from __future__ import annotations

import csv
import hashlib
import pickle
import random
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

from .features import smiles_to_graph


DATASET_CONFIGS = {
    "bace": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv",
        ],
        "md5": "ba7f8fa3fdf463a811fa7edea8c982c2",
        "file_name": "bace.csv",
        "smiles_field": "mol",
        "label_field": "Class",
    },
    "hiv": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
            "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv",
        ],
        "md5": "9ad10c88f82f1dac7eb5c52b668c30a7",
        "file_name": "HIV.csv",
        "smiles_field": "smiles",
        "label_field": "HIV_active",
    },
}


class MoleculeDataset:
    def __init__(self, name: str, graphs: Sequence[Dict[str, np.ndarray]]):
        self.name = name
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return self.graphs[index]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([float(item["label"][0]) for item in self.graphs], dtype=np.float32)

    @property
    def smiles(self) -> List[str]:
        return [str(item["smiles"]) for item in self.graphs]

    @property
    def node_feature_dim(self) -> int:
        return int(self.graphs[0]["node_feature"].shape[-1])


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(name: str, data_dir: Path) -> Path:
    config = DATASET_CONFIGS[name]
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / config["file_name"]
    if csv_path.exists() and md5sum(csv_path) == config["md5"]:
        return csv_path

    last_error = None
    for url in config["urls"]:
        try:
            print(f"Downloading {name} from {url}")
            urllib.request.urlretrieve(url, csv_path)
            if md5sum(csv_path) != config["md5"]:
                raise RuntimeError(f"MD5 mismatch for {csv_path}")
            return csv_path
        except Exception as exc:
            last_error = exc
            if csv_path.exists():
                csv_path.unlink()
    raise RuntimeError(f"Failed to download {name}: {last_error}")


def load_dataset(name: str, data_dir: Path, use_cache: bool = True) -> MoleculeDataset:
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset `{name}`. Available: {sorted(DATASET_CONFIGS)}")

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / f"{name}_graphs.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fin:
            graphs = pickle.load(fin)
        return MoleculeDataset(name, graphs)

    config = DATASET_CONFIGS[name]
    csv_path = download_dataset(name, data_dir)
    graphs = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            smiles = row[config["smiles_field"]]
            label_text = row[config["label_field"]]
            if label_text == "":
                skipped += 1
                continue
            graph = smiles_to_graph(smiles, float(label_text))
            if graph is None:
                skipped += 1
                continue
            graphs.append(graph)

    if use_cache:
        with cache_path.open("wb") as fout:
            pickle.dump(graphs, fout)
    print(f"Loaded {name}: {len(graphs)} molecules, skipped {skipped}")
    return MoleculeDataset(name, graphs)


def subset(dataset: MoleculeDataset, indices: Iterable[int]) -> MoleculeDataset:
    return MoleculeDataset(dataset.name, [dataset.graphs[i] for i in indices])


def random_split(
    dataset: MoleculeDataset,
    seed: int,
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(fractions[0] * len(indices))
    valid_end = train_end + int(fractions[1] * len(indices))
    return (
        subset(dataset, indices[:train_end]),
        subset(dataset, indices[train_end:valid_end]),
        subset(dataset, indices[valid_end:]),
    )


def scaffold_split(
    dataset: MoleculeDataset,
    seed: int,
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    scaffold_to_indices: Dict[str, List[int]] = {}
    for index, smiles in enumerate(dataset.smiles):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
        scaffold_to_indices.setdefault(scaffold, []).append(index)

    rng = random.Random(seed)
    scaffold_sets = list(scaffold_to_indices.values())
    for scaffold_set in scaffold_sets:
        rng.shuffle(scaffold_set)
    scaffold_sets.sort(key=lambda item: (len(item), item[0]), reverse=True)

    train_cutoff = fractions[0] * len(dataset)
    valid_cutoff = (fractions[0] + fractions[1]) * len(dataset)
    train_indices, valid_indices, test_indices = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(valid_indices) + len(scaffold_set) <= valid_cutoff:
            valid_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    return subset(dataset, train_indices), subset(dataset, valid_indices), subset(dataset, test_indices)


def split_dataset(
    dataset: MoleculeDataset,
    split: str,
    seed: int,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    if split == "random":
        return random_split(dataset, seed)
    if split == "scaffold":
        return scaffold_split(dataset, seed)
    raise ValueError(f"Unknown split `{split}`")


def collate_batch(graphs: Sequence[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(graphs)
    max_node = max(int(item["node_feature"].shape[0]) for item in graphs)
    feature_dim = int(graphs[0]["node_feature"].shape[-1])
    node_feature = np.zeros((batch_size, max_node, feature_dim), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_node, max_node), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_node), dtype=np.float32)
    label = np.zeros((batch_size, 1), dtype=np.float32)

    for i, item in enumerate(graphs):
        num_node = int(item["node_feature"].shape[0])
        node_feature[i, :num_node] = item["node_feature"]
        adjacency[i, :num_node, :num_node] = item["adjacency"]
        node_mask[i, :num_node] = 1.0
        label[i] = item["label"]
    return node_feature, adjacency, node_mask, label


def batch_iterator(
    dataset: MoleculeDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield collate_batch([dataset[i] for i in batch_indices])
