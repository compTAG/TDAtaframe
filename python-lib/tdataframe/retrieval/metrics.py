from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_1d_array(values: Sequence[float]) -> np.ndarray:
    """Convert a vector-like input to a dense 1D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}")
    return arr


def l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Euclidean distance between two descriptor vectors."""
    left_arr = as_1d_array(left)
    right_arr = as_1d_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Vector shape mismatch for l2 distance: {left_arr.shape} vs {right_arr.shape}"
        )
    return float(np.linalg.norm(left_arr - right_arr))


def l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the L1 distance between two descriptor vectors."""
    left_arr = as_1d_array(left)
    right_arr = as_1d_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Vector shape mismatch for l1 distance: {left_arr.shape} vs {right_arr.shape}"
        )
    return float(np.abs(left_arr - right_arr).sum())


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine distance, defined as 1 - cosine similarity."""
    left_arr = as_1d_array(left)
    right_arr = as_1d_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Vector shape mismatch for cosine distance: {left_arr.shape} vs {right_arr.shape}"
        )

    left_norm = np.linalg.norm(left_arr)
    right_norm = np.linalg.norm(right_arr)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine distance is undefined for zero-norm vectors")

    similarity = float(np.dot(left_arr, right_arr) / (left_norm * right_norm))
    return 1.0 - similarity
