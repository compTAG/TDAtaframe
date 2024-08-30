from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr
from pathlib import Path
import polars as pl
from typing import Optional

lib = Path(__file__).parent.parent


def pre_align_wect(
    vertices: IntoExpr,
    triangles: IntoExpr,
    weights: IntoExpr,
    *,
    embedded_dimension: int,
    num_heights: int,
    num_directions: int,
    subsample_ratio: float,
    subsample_min: int,
    subsample_max: int,
    eps: Optional[float] = None,
    copies: bool,
):
    return register_plugin_function(
        args=[vertices, triangles, weights],
        plugin_path=lib,
        function_name="premapped_wect3",
        is_elementwise=True,
        kwargs=dict(
            embedded_dimension=embedded_dimension,
            num_heights=num_heights,
            num_directions=num_directions,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            subsample_max=subsample_max,
            eps=eps,
            copies=copies,
        ),
    )
