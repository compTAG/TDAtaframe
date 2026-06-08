from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Literal

import numpy as np
import polars as pl

MetricName = Literal["l1", "l2", "cosine"]


def _as_1d_array(values: Sequence[float]) -> np.ndarray:
    """Convert a vector-like input to a dense 1D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}")
    return arr


def l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Euclidean distance between two descriptor vectors."""
    left_arr = _as_1d_array(left)
    right_arr = _as_1d_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Vector shape mismatch for l2 distance: {left_arr.shape} vs {right_arr.shape}"
        )
    return float(np.linalg.norm(left_arr - right_arr))


def l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the L1 distance between two descriptor vectors."""
    left_arr = _as_1d_array(left)
    right_arr = _as_1d_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Vector shape mismatch for l1 distance: {left_arr.shape} vs {right_arr.shape}"
        )
    return float(np.abs(left_arr - right_arr).sum())


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine distance, defined as 1 - cosine similarity."""
    left_arr = _as_1d_array(left)
    right_arr = _as_1d_array(right)
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


def _metric_function(metric: MetricName) -> Callable[[Sequence[float], Sequence[float]], float]:
    if metric == "l2":
        return l2_distance
    if metric == "l1":
        return l1_distance
    if metric == "cosine":
        return cosine_distance
    raise ValueError(f"Unsupported metric {metric!r}")


def exact_rank(
    df: pl.DataFrame | pl.LazyFrame,
    query: Sequence[float],
    descriptor_column: str,
    *,
    metric: MetricName = "l2",
    score_column: str = "distance",
) -> pl.LazyFrame:
    """Score every row in a descriptor table against one query vector.

    This is the simplest retrieval baseline for `tdataframe`: exact scan over
    one descriptor vector per row, followed by sorting by ascending distance.
    """
    query_arr = _as_1d_array(query)
    metric_fn = _metric_function(metric)

    def score_descriptor(values: Sequence[float]) -> float:
        return metric_fn(query_arr, values)

    return (
        df.lazy()
        .with_columns(
            pl.col(descriptor_column).map_elements(
                score_descriptor,
                return_dtype=pl.Float64,
            ).alias(score_column)
        )
        .sort(score_column)
    )


def exact_knn(
    df: pl.DataFrame | pl.LazyFrame,
    query: Sequence[float],
    descriptor_column: str,
    *,
    k: int,
    metric: MetricName = "l2",
    score_column: str = "distance",
) -> pl.DataFrame:
    """Return the top-k nearest rows to one query descriptor."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    return exact_rank(
        df,
        query,
        descriptor_column,
        metric=metric,
        score_column=score_column,
    ).head(k).collect()
