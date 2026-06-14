"""Check dependencies for the MindSpore molecular GNN reproduction."""

from __future__ import annotations

import importlib
import platform
import sys


MODULES = {
    "mindspore": "mindspore",
    "rdkit": "rdkit",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "notebook": "notebook",
}


def version_of(module) -> str:
    return getattr(module, "__version__", "installed")


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
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
    if missing:
        print("Environment is not ready.")
        print("Install dependencies with the commands in README.md.")
    else:
        print("Environment looks ready.")
        print("Try: python run_experiment.py --dataset bace --model gin --epoch 1 --batch_size 32 --seed 0")


if __name__ == "__main__":
    main()
