from __future__ import annotations

from typing import Callable, Literal

import polars as pl

from .metrics import as_1d_array, cosine_distance, l1_distance, l2_distance

MetricName = Literal["l1", "l2", "cosine"]


def _metric_function(metric: MetricName) -> Callable[[list[float], list[float]], float]:
    if metric == "l2":
        return l2_distance
    if metric == "l1":
        return l1_distance
    if metric == "cosine":
        return cosine_distance
    raise ValueError(f"Unsupported metric {metric!r}")


def exact_rank(
    df: pl.DataFrame | pl.LazyFrame,
    query: list[float],
    descriptor_column: str,
    *,
    metric: MetricName = "l2",
    score_column: str = "distance",
) -> pl.LazyFrame:
    """Score every row in a descriptor table against one query vector.

    This is the simplest retrieval baseline for `tdataframe`: exact scan over
    one descriptor vector per row, followed by sorting by ascending distance.
    """
    query_arr = as_1d_array(query)
    metric_fn = _metric_function(metric)

    def score_descriptor(values: list[float]) -> float:
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
    query: list[float],
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
