"""Implement queries for polars dataframes to be used in shape-matching.

Specifically, these queries support the following pipeline:
    - Compute pre-alignment of mesh
    - Compute WECT on the pre-aligned mesh
    - Compute an embedding/post-processing of the WECT.

This whole pipeline can be called using compute_db_entry().
"""

# TODO: DOCSTRINGS

from typing import List
import polars as pl
from ..utils import flatten_matrix, unflatten_to_matrix
from ..params import WeightedComplexInfo, MapArgs, WectArgs
from .register import pre_align_wect, wect


def premapped_wects(
    wci: WeightedComplexInfo,
    ma: MapArgs,
    wa: WectArgs,
) -> pl.Expr:
    """
    Compute the WECTs for the given simplices and weights.
    The simplices and weights columns are each flattened structs.
    """

    wects = pre_align_wect(
        pl.col(wci.simplices),
        pl.col(wci.weights),
        provided_simplices=wci.provided_simplices,
        provided_weights=wci.provided_weights,
        embedded_dimension=wci.vdim,
        num_heights=wa.steps,
        num_directions=wa.directions,
        align_dimension=ma.align_dimension,
        subsample_ratio=ma.subsample_ratio,
        subsample_min=ma.subsample_min,
        subsample_max=ma.subsample_max,
        eps=ma.eps,
        copies=ma.copies,
    )  # output column of (n * d * d) flattened array of flattened matrices

    # Reshape the maps
    wects = unflatten_to_matrix(
        wects, wa.steps * wa.directions
    )  # now (n)-list of num_height * num_direction flattened matrices

    return wects


# def _get_embedding_expr(
#     veccol: pl.Expr,
#     embedder: Embedder,
# ) -> pl.Expr:
#     # obtain call to gufunc so polars recognizes signature
#     emb_gufunc = embedder.get_gufunc()
#     args = embedder.get_gufunc_args()
#     return veccol.map_batches(
#         lambda t: emb_gufunc(t, *args),
#         return_dtype=pl.array(pl.float64, embedder.emb_dim),
#         is_elementwise=true,
#     )


def with_premapped_wects(
    df: pl.LazyFrame | pl.DataFrame,
    wci: WeightedComplexInfo,
    ma: MapArgs,
    wa: WectArgs,
    wname: str,
) -> pl.LazyFrame:
    """Compute the WECTs for the given mesh data and transofrmations.

    provided simplices and weights need to be in order.
    leave out 0 from provided weights TODO: parse this? or do in rust backend?

    TODO: Verify the rest of this docstring.

    Returns:
        A lazyframe with the computed WECTs. The rows correspond to each WECT
        given by a transformation on an object.

    """

    return df.lazy().with_columns(premapped_wects(wci, ma, wa).alias(wname))


def wects(
    wci: WeightedComplexInfo,
    wa: WectArgs,
) -> pl.Expr:
    """
    Compute the WECTs for the given simplices and weights.
    The simplices and weights columns are each flattened structs.
    """

    wects = wect(
        pl.col(wci.simplices),
        pl.col(wci.weights),
        provided_simplices=wci.provided_simplices,
        provided_weights=wci.provided_weights,
        embedded_dimension=wci.vdim,
        num_heights=wa.steps,
        num_directions=wa.directions,
    )  # output column of (n * d * d) flattened array of flattened matrices

    # Reshape the maps
    wects = unflatten_to_matrix(
        wects, wa.steps * wa.directions
    )  # now (n)-list of num_height * num_direction flattened matrices

    return wects


# def _get_embedding_expr(
#     veccol: pl.Expr,
#     embedder: Embedder,
# ) -> pl.Expr:
#     # obtain call to gufunc so polars recognizes signature
#     emb_gufunc = embedder.get_gufunc()
#     args = embedder.get_gufunc_args()
#     return veccol.map_batches(
#         lambda t: emb_gufunc(t, *args),
#         return_dtype=pl.array(pl.float64, embedder.emb_dim),
#         is_elementwise=true,
#     )


def with_wects(
    df: pl.LazyFrame | pl.DataFrame,
    wci: WeightedComplexInfo,
    wa: WectArgs,
    wname: str,
) -> pl.LazyFrame:
    """Compute the WECTs for the given mesh data and transofrmations.

    provided simplices and weights need to be in order.
    leave out 0 from provided weights TODO: parse this? or do in rust backend?

    TODO: Verify the rest of this docstring.

    Returns:
        A lazyframe with the computed WECTs. The rows correspond to each WECT
        given by a transformation on an object.

    """

    return df.lazy().with_columns(wects(wci, wa).alias(wname))
