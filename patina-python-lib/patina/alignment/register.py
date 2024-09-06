from typing import Optional
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

lib = Path(__file__).parent.parent


def barycenters(
    vertices: IntoExpr,
    simplices: IntoExpr,
    *,
    embedded_dimension: int,
    simplex_dimension: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[vertices, simplices],
        plugin_path=lib,
        function_name="barycenters",
        is_elementwise=True,
        kwargs=dict(
            embedded_dimension=embedded_dimension, simplex_dimension=simplex_dimension
        ),
    )


def maps_svd_copies(
    vertices: IntoExpr,
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    embedded_dimension: int,
    simplex_dimension: int,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
    eps: Optional[float] = None,
    copies: bool,
):
    return register_plugin_function(
        args=[vertices, simplices, weights],
        plugin_path=lib,
        function_name="maps_svd_copies",
        is_elementwise=True,
        kwargs=dict(
            embedded_dimension=embedded_dimension,
            simplex_dimension=simplex_dimension,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
            eps=eps,
            copies=copies,
        ),
    )


def map_svd(
    vertices: IntoExpr,
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    embedded_dimension: int,
    simplex_dimension: int,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
):
    return register_plugin_function(
        args=[vertices, simplices, weights],
        plugin_path=lib,
        function_name="map_svd",
        is_elementwise=True,
        kwargs=dict(
            embedded_dimension=embedded_dimension,
            simplex_dimension=simplex_dimension,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
        ),
    )
