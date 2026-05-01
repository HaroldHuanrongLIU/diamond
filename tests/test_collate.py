from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diamond_surgwmbench.data import (
    SurgWMBenchClipDataset,
    SurgWMBenchFrameDataset,
    collate_dense_variable_length,
    collate_frame_autoencoding,
    collate_sparse_anchors,
    collate_transition_pairs,
    collate_window_sequences,
)


def test_sparse_collate_returns_expected_shapes(toy_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(toy_root, "manifests/train.jsonl", frame_sampling="sparse_anchors", image_size=32)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_sparse_anchors)))

    assert batch["frames"].shape == (2, 20, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 20, 2)
    assert batch["coords_px"].shape == (2, 20, 2)
    assert batch["sampled_indices"].shape == (2, 20)
    assert batch["human_anchor_mask"].all()
    assert batch["anchor_dt"].shape == (2, 19)
    assert batch["actions_delta"].shape == (2, 19, 2)
    assert batch["actions_delta_dt"].shape == (2, 19, 3)
    assert batch["direction_classes"].shape == (2, 19)
    assert batch["magnitudes"].shape == (2, 19)
    expected_dt = (batch["sampled_indices"][:, 1:] - batch["sampled_indices"][:, :-1]).float()
    expected_dt = expected_dt / (batch["num_frames"].float().view(-1, 1) - 1.0)
    assert torch.allclose(batch["actions_delta_dt"][..., 2], expected_dt)


def test_dense_collate_pads_and_masks(toy_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_root,
        "manifests/train.jsonl",
        frame_sampling="dense",
        image_size=32,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_dense_variable_length)))

    assert batch["frames"].shape == (2, 31, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 31, 2)
    assert batch["coords_px"].shape == (2, 31, 2)
    assert batch["frame_mask"][0, :25].all()
    assert not batch["frame_mask"][0, 25:].any()
    assert batch["frame_mask"][1, :31].all()
    assert batch["frame_indices"][0, 25:].eq(-1).all()
    assert batch["actions_delta"].shape == (2, 30, 2)
    assert batch["actions_delta_dt"].shape == (2, 30, 3)
    assert batch["action_mask"].shape == (2, 30)
    assert not batch["action_mask"][0, 24:].any()


def test_transition_pair_collate_returns_pair_contract(toy_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_root,
        "manifests/train.jsonl",
        frame_sampling="transition_pairs",
        image_size=32,
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_transition_pairs)))

    assert batch["frame_t"].shape == (4, 3, 32, 32)
    assert batch["frame_tp1"].shape == (4, 3, 32, 32)
    assert batch["coord_t"].shape == (4, 2)
    assert batch["coord_tp1"].shape == (4, 2)
    assert batch["action_delta_dt"].shape == (4, 3)
    assert batch["coord_source_t"].shape == (4,)
    assert batch["coord_source_tp1"].shape == (4,)
    assert batch["label_weight_tp1"].shape == (4,)


def test_window_collate_uses_dense_padding_contract(toy_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_root,
        "manifests/train.jsonl",
        frame_sampling="window",
        max_frames=10,
        image_size=16,
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_window_sequences)))

    assert batch["frames"].shape[:2] == (4, 10)
    assert batch["frame_mask"].shape == (4, 10)
    assert batch["actions_delta_dt"].shape == (4, 9, 3)


def test_frame_autoencoding_collate(toy_root: Path) -> None:
    dataset = SurgWMBenchFrameDataset(toy_root, "manifests/train.jsonl", image_size=20)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_frame_autoencoding)))

    assert batch["images"].shape == (3, 3, 20, 20)
    assert batch["frames"].shape == (3, 3, 20, 20)
    assert len(batch["metadata"]) == 3
