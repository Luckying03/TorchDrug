"""RDKit-based molecular graph features for the MindSpore reproduction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem


ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


def one_hot_with_unknown(value, choices):
    """One-hot encode value; the final slot means unknown."""
    encoding = [int(value == choice) for choice in choices]
    encoding.append(int(value not in choices))
    return encoding


def atom_to_feature(atom: rdchem.Atom) -> np.ndarray:
    feature = []
    feature += one_hot_with_unknown(atom.GetAtomicNum(), ATOM_TYPES)
    feature += one_hot_with_unknown(atom.GetTotalDegree(), DEGREES)
    feature += one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
    feature += one_hot_with_unknown(atom.GetTotalNumHs(), NUM_HS)
    feature += one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
    feature += [
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass() * 0.01,
    ]
    return np.asarray(feature, dtype=np.float32)


def smiles_to_graph(smiles: str, label: float) -> Optional[Dict[str, np.ndarray]]:
    """Convert a SMILES string into node features and a dense adjacency matrix."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    node_feature = np.stack([atom_to_feature(atom) for atom in mol.GetAtoms()], axis=0)
    num_atom = mol.GetNumAtoms()
    adjacency = np.zeros((num_atom, num_atom), dtype=np.float32)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0

    return {
        "smiles": smiles,
        "node_feature": node_feature,
        "adjacency": adjacency,
        "label": np.asarray([label], dtype=np.float32),
        "num_node": np.asarray([num_atom], dtype=np.int32),
    }


def feature_dim() -> int:
    dummy = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
    return int(atom_to_feature(dummy).shape[0])
