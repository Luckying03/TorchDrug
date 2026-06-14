"""Check whether the current Python environment can run this TorchDrug project."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "README.md").exists() else SCRIPT_DIR.parent
TORCHDRUG_CANDIDATES = [
    PROJECT_ROOT / "torchdrug",
    PROJECT_ROOT.parent / "torchdrug",
]
LOCAL_TORCHDRUG = None
for local_torchdrug in TORCHDRUG_CANDIDATES:
    if (local_torchdrug / "torchdrug" / "__init__.py").exists():
        LOCAL_TORCHDRUG = local_torchdrug
        sys.path.insert(0, str(local_torchdrug))
        break


MODULES = {
    "torch": "torch",
    "torch_scatter": "torch-scatter",
    "torch_cluster": "torch-cluster",
    "rdkit": "rdkit",
    "decorator": "decorator",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "networkx": "networkx",
    "ninja": "ninja",
    "jinja2": "jinja2",
    "lmdb": "lmdb",
    "esm": "fair-esm",
}


def version_of(module) -> str:
    return getattr(module, "__version__", "installed")


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Project root:", PROJECT_ROOT)
    print("Local TorchDrug source:", LOCAL_TORCHDRUG or "not found")
    print()

    py_ok = (3, 7) <= sys.version_info[:2] <= (3, 10)
    print("Python compatibility:", "OK" if py_ok else "NOT OK (TorchDrug declares >=3.7,<3.11)")
    print()

    missing = []
    for import_name, package_name in MODULES.items():
        try:
            module = importlib.import_module(import_name)
            print(f"[OK] {package_name}: {version_of(module)}")
        except Exception as exc:
            missing.append(package_name)
            print(f"[MISSING] {package_name}: {type(exc).__name__}: {exc}")

    print()
    try:
        from torchdrug import core, datasets, models, tasks  # noqa: F401

        print("[OK] TorchDrug high-level imports: core, datasets, models, tasks")
    except Exception as exc:
        print("[FAIL] TorchDrug high-level imports")
        print(f"       {type(exc).__name__}: {exc}")

    print()
    if missing or not py_ok:
        print("Recommended fix:")
        print("  Create a fresh conda environment with Python 3.10 and install TorchDrug dependencies.")
        print("  See README.md for exact commands.")
    else:
        print("Environment looks ready. Try:")
        print("  python experiments/run_experiment.py --dataset bace --model gin --epoch 1 --batch_size 128 --seed 0")


if __name__ == "__main__":
    main()
