from typing import List, Optional
from pathlib import Path

from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

lib = Path(__file__).parent.parent


def pre_align_copy_wect(
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    num_heights: int,
    num_directions: int,
    provided_weights: List[int],
    align_dimension: int,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
    eps: Optional[float] = None,
    copies: bool,
):
    return register_plugin_function(
        args=[simplices, weights],
        plugin_path=lib,
        function_name="premapped_copy_wect",
        is_elementwise=True,
        kwargs=dict(
            num_heights=num_heights,
            num_directions=num_directions,
            provided_weights=provided_weights,
            align_dimension=align_dimension,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
            eps=eps,
            copies=copies,
        ),
    )


def pre_align_wect(
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    num_heights: int,
    num_directions: int,
    provided_weights: List[int],
    align_dimension: int,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
):
    return register_plugin_function(
        args=[simplices, weights],
        plugin_path=lib,
        function_name="premapped_wect",
        is_elementwise=True,
        kwargs=dict(
            num_heights=num_heights,
            num_directions=num_directions,
            provided_weights=provided_weights,
            align_dimension=align_dimension,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
        ),
    )


def wect(
    simplices: IntoExpr,
    weights: IntoExpr,
    *,
    num_heights: int,
    num_directions: int,
    provided_weights: List[int],
):
    return register_plugin_function(
        args=[simplices, weights],
        plugin_path=lib,
        function_name="wect",
        is_elementwise=True,
        kwargs=dict(
            num_heights=num_heights,
            num_directions=num_directions,
            provided_weights=provided_weights,
        ),
    )


def ect(
    simplices: IntoExpr,
    *,
    num_directions: int,
    num_heights: int,
):
    return register_plugin_function(
        args=[simplices],
        plugin_path=lib,
        function_name="ect",
        is_elementwise=True,
        kwargs=dict(
            num_directions=num_directions,
            num_heights=num_heights,
        ),
    )
