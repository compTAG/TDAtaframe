from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr
from pathlib import Path
import polars as pl
from typing import Optional

lib = Path(__file__).parent


def barycenters(vertices: IntoExpr, simplices: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[vertices, simplices],
        plugin_path=lib,
        function_name="barycenters",
        is_elementwise=True,
    )


def maps_svd(
    vertices: IntoExpr,
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
    eps: Optional[float] = None,
    copies: bool,
):
    return register_plugin_function(
        args=[vertices, simplices, weights],
        plugin_path=lib,
        function_name="maps_svd",
        is_elementwise=True,
        kwargs=dict(
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
            eps=eps,
            copies=copies,
        ),
    )
