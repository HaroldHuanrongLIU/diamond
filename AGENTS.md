# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the original DIAMOND Atari code: agents, replay data, Atari env wrappers, diffusion models, training, and play utilities.
- `config/` contains Hydra configs for DIAMOND (`config/trainer.yaml`, `config/env/atari.yaml`, `config/agent/default.yaml`).
- `diamond_surgwmbench/` is the SurgWMBench extension package. Current first-pass code covers data loading, raw video/frame loading, collators, transforms, and trajectory metrics.
- `tools/` contains runnable utility modules, including toy SurgWMBench generation and loader validation.
- `tests/` contains pytest coverage for the SurgWMBench data foundation.
- `results/` stores static result data from the original DIAMOND repo.

## Build, Test, and Development Commands

- Install dependencies: `pip install -r requirements.txt`.
- Run original DIAMOND Atari training: `python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0`.
- Play or inspect pretrained DIAMOND models: `python src/play.py --pretrained`.
- Validate real SurgWMBench data:
  `python -m tools.validate_surgwmbench_loader --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench --manifest manifests/train.jsonl --interpolation-method linear --check-files --num-samples 8`.
- Generate toy SurgWMBench data:
  `python -m tools.make_toy_surgwmbench --output /tmp/SurgWMBench --num-clips 2`.
- Run focused SurgWMBench tests:
  `pytest -q -p no:cacheprovider tests/test_surgwmbench_dataset.py tests/test_collate.py tests/test_metrics.py tests/test_validate_surgwmbench_loader.py`.

## Coding Style & Naming Conventions

Use Python 3.10+ style with type hints, dataclasses where useful, and `pathlib.Path` for filesystem paths. Prefer explicit names such as `sampled_indices`, `local_frame_idx`, and `human_anchor_coords_px`. Keep original DIAMOND `src/` code readable; add SurgWMBench-specific work under `diamond_surgwmbench/` unless a deliberate integration change is required.

## Testing Guidelines

Tests use pytest and should be named `test_*.py`. New data code should be covered with toy datasets created under `tmp_path`; do not depend on the real dataset for unit tests. Real-data checks belong in smoke commands or documentation. Preserve the separation between sparse human labels and dense pseudo coordinates in tests and assertions.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Add SurgWMBench data foundation` and `Update requirements to support Python 3.11+`. Keep commits scoped and avoid mixing original DIAMOND refactors with SurgWMBench additions. PRs should describe what changed, list validation commands run, and call out any dataset-path or CUDA assumptions.

## Agent-Specific Instructions

Do not create random train/val/test splits for SurgWMBench. Use official manifests only. Do not treat dense pseudo coordinates as human ground truth, do not infer difficulty from paths, and do not modify dataset annotations.
