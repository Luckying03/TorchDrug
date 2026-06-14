"""RDKit molecular features aligned with TorchDrug defaults."""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
from rdkit import Chem


# These vocabularies mirror torchdrug/torchdrug/data/feature.py.
ATOM_VOCAB = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
ATOM_VOCAB = {atom: i for i, atom in enumerate(ATOM_VOCAB)}
DEGREE_VOCAB = range(7)
NUM_HS_VOCAB = range(7)
FORMAL_CHARGE_VOCAB = range(-5, 6)
CHIRAL_TAG_VOCAB = range(4)
NUM_RADICAL_VOCAB = range(8)
HYBRIDIZATION_VOCAB = range(len(Chem.rdchem.HybridizationType.values))

BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPE_VOCAB = {bond_type: i for i, bond_type in enumerate(BOND_TYPE_VOCAB)}
BOND_DIR_VOCAB = range(len(Chem.rdchem.BondDir.values))
BOND_STEREO_VOCAB = range(len(Chem.rdchem.BondStereo.values))


def _as_int(value):
    try:
        return int(value)
    except TypeError:
        return value


def onehot(value, vocab, allow_unknown: bool = False):
    """TorchDrug-style one-hot encoding."""
    if isinstance(vocab, dict):
        index = vocab[value] if value in vocab else -1
    else:
        value = _as_int(value)
        vocab = list(vocab)
        index = vocab.index(value) if value in vocab else -1

    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn(f"Unknown value `{value}`", RuntimeWarning)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(f"Unknown value `{value}`. Available vocabulary is `{vocab}`")
        feature[index] = 1
    return feature


def atom_default(atom: Chem.rdchem.Atom) -> np.ndarray:
    """Replicate TorchDrug's default atom feature (69 dimensions)."""
    feature = (
        onehot(atom.GetSymbol(), ATOM_VOCAB, allow_unknown=True)
        + onehot(atom.GetChiralTag(), CHIRAL_TAG_VOCAB)
        + onehot(atom.GetTotalDegree(), DEGREE_VOCAB, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), FORMAL_CHARGE_VOCAB)
        + onehot(atom.GetTotalNumHs(), NUM_HS_VOCAB)
        + onehot(atom.GetNumRadicalElectrons(), NUM_RADICAL_VOCAB)
        + onehot(atom.GetHybridization(), HYBRIDIZATION_VOCAB)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )
    return np.asarray(feature, dtype=np.float32)


def atom_simple(atom: Chem.rdchem.Atom) -> np.ndarray:
    """The earlier compact feature used for the simple baseline."""
    atom_types = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
    degrees = [0, 1, 2, 3, 4, 5]
    charges = [-2, -1, 0, 1, 2]
    hydrogens = [0, 1, 2, 3, 4]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    feature = (
        _onehot_unknown(atom.GetAtomicNum(), atom_types)
        + _onehot_unknown(atom.GetTotalDegree(), degrees)
        + _onehot_unknown(atom.GetFormalCharge(), charges)
        + _onehot_unknown(atom.GetTotalNumHs(), hydrogens)
        + _onehot_unknown(atom.GetHybridization(), hybridizations)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing()), atom.GetMass() * 0.01]
    )
    return np.asarray(feature, dtype=np.float32)


def _onehot_unknown(value, choices):
    encoding = [int(value == choice) for choice in choices]
    encoding.append(int(value not in choices))
    return encoding


def bond_default(bond: Chem.rdchem.Bond) -> np.ndarray:
    """Replicate TorchDrug's default bond feature (19 dimensions)."""
    feature = (
        onehot(bond.GetBondType(), BOND_TYPE_VOCAB)
        + onehot(bond.GetBondDir(), BOND_DIR_VOCAB)
        + onehot(bond.GetStereo(), BOND_STEREO_VOCAB)
        + [int(bond.GetIsConjugated())]
    )
    return np.asarray(feature, dtype=np.float32)


def smiles_to_graph(smiles: str, label: float, feature_set: str = "torchdrug_default") -> Optional[Dict[str, np.ndarray]]:
    """Convert a SMILES string into TorchDrug-style edge-list graph tensors.

    `feature_set="torchdrug_default"` mirrors TorchDrug's default atom/bond
    features. `feature_set="simple"` keeps the earlier compact baseline.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_fn = atom_simple if feature_set == "simple" else atom_default
    node_feature = np.stack([atom_fn(atom) for atom in mol.GetAtoms()], axis=0)
    num_atom = mol.GetNumAtoms()
    edge_list = []
    edge_feature = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_list += [[begin, end], [end, begin]]
        if feature_set == "torchdrug_default":
            feature = bond_default(bond)
        else:
            feature = np.zeros((bond_feature_dim(feature_set),), dtype=np.float32)
        edge_feature += [feature, feature]

    if edge_list:
        edge_list = np.asarray(edge_list, dtype=np.int32)
        edge_feature = np.asarray(edge_feature, dtype=np.float32)
    else:
        edge_list = np.zeros((0, 2), dtype=np.int32)
        edge_feature = np.zeros((0, bond_feature_dim(feature_set)), dtype=np.float32)

    return {
        "smiles": smiles,
        "node_feature": node_feature,
        "edge_list": edge_list,
        "edge_feature": edge_feature,
        "label": np.asarray([label], dtype=np.float32),
        "num_node": np.asarray([num_atom], dtype=np.int32),
        "num_edge": np.asarray([len(edge_list)], dtype=np.int32),
    }


def node_feature_dim(feature_set: str = "torchdrug_default") -> int:
    dummy = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
    atom_fn = atom_simple if feature_set == "simple" else atom_default
    return int(atom_fn(dummy).shape[0])


def bond_feature_dim(feature_set: str = "torchdrug_default") -> int:
    if feature_set == "simple":
        return 1
    dummy = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
    return int(bond_default(dummy).shape[0])
