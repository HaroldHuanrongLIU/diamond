from __future__ import annotations

import json

import numpy as np
import torch

from diamond_surgwmbench.evaluation.metrics import (
    ade,
    discrete_frechet,
    endpoint_error,
    error_by_horizon,
    fde,
    symmetric_hausdorff,
    trajectory_length,
    trajectory_length_error,
    trajectory_smoothness,
)


def test_metrics_match_simple_shifted_trajectories() -> None:
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = target + torch.tensor([0.0, 1.0])

    assert ade(pred, target) == 1.0
    assert fde(pred, target) == 1.0
    assert endpoint_error(pred, target) == 1.0
    assert discrete_frechet(pred, target) == 1.0
    assert symmetric_hausdorff(pred, target) == 1.0
    assert trajectory_length(pred) == 2.0
    assert trajectory_length_error(pred, target) == 0.0
    assert trajectory_smoothness(pred) == 0.0
    assert error_by_horizon(pred, target, [1, 2, 3]) == {1: 1.0, 2: 1.0, 3: 1.0}


def test_metrics_support_batches_masks_numpy_and_json_values() -> None:
    target = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        ]
    )
    pred = target.copy()
    pred[0, :, 0] += 1.0
    pred[1, :, 1] += 2.0
    mask = np.array([[True, True, True], [True, False, False]])

    assert ade(pred, target, mask) == 1.25
    assert fde(pred, target, mask) == 1.5
    horizons = error_by_horizon(pred, target, [1, 3], mask)
    assert horizons[1] == 1.5
    assert horizons[3] == 1.0
    json.dumps({"ade": ade(pred, target, mask), "horizons": horizons})


def test_metrics_return_none_for_empty_masks() -> None:
    coords = torch.zeros(2, 3, 2)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    assert ade(coords, coords, mask) is None
    assert fde(coords, coords, mask) is None
    assert trajectory_length(coords, mask) is None
    assert trajectory_smoothness(coords, mask) is None
