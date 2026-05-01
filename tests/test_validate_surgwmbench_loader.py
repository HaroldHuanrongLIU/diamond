from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.validate_surgwmbench_loader import validate_surgwmbench


def test_validate_surgwmbench_loader_passes_on_toy_data(toy_root: Path) -> None:
    errors = validate_surgwmbench(
        dataset_root=toy_root,
        manifest="manifests/train.jsonl",
        interpolation_method="linear",
        check_files=True,
        num_samples=2,
    )

    assert errors == []


def test_validate_surgwmbench_loader_module_command(toy_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.validate_surgwmbench_loader",
            "--dataset-root",
            str(toy_root),
            "--manifest",
            "manifests/train.jsonl",
            "--interpolation-method",
            "linear",
            "--check-files",
            "--num-samples",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "SurgWMBench validation passed." in result.stdout


def test_new_code_contains_no_random_split_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_paths = [root / "diamond_surgwmbench", root / "tools"]
    text = "\n".join(path.read_text(encoding="utf-8") for base in checked_paths for path in base.rglob("*.py"))

    assert "random_split" not in text
    assert "train_test_split" not in text
