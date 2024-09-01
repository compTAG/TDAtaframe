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
from .register import pre_align_wect_3D_triangles, pre_align_wect


def get_premapped_wect_3D_triangles_expr(
    vdim, vcol, tcol, ncol, eps, rot_params, wect_params
):  # noqa
    num_heights = wect_params["steps"]
    num_directions = wect_params["directions"]
    wects = pre_align_wect_3D_triangles(
        flatten_matrix(pl.col(vcol)),
        flatten_matrix(pl.col(tcol)),
        flatten_matrix(pl.col(ncol)),
        embedded_dimension=vdim,
        num_heights=num_heights,
        num_directions=num_directions,
        subsample_ratio=rot_params["subsample_ratio"],
        subsample_min=rot_params["subsample_min"],
        subsample_max=rot_params["subsample_max"],
        eps=eps,
        copies=rot_params["copies"],
    )  # output column of (n * d * d) flattened array of flattened matrices

    # Reshape the maps
    wects = unflatten_to_matrix(
        wects, num_heights * num_directions
    )  # now (n)-list of num_height * num_direction flattened matrices

    return wects


def get_premapped_wect_expr(
    vdim, scol, wcol, provided_simplices, provided_weights, rot_params, wect_params
):  # noqa
    """
    Compute the WECTs for the given simplices and weights.
    The simplices and weights columns are each flattened structs.
    """
    num_heights = wect_params["steps"]
    num_directions = wect_params["directions"]

    eps = None
    if rot_params["heur_fix"]:
        eps = rot_params["eps"]

    wects = pre_align_wect(
        pl.col(scol),
        pl.col(wcol),
        provided_simplices=provided_simplices,
        provided_weights=provided_weights,
        embedded_dimension=vdim,
        num_heights=num_heights,
        num_directions=num_directions,
        subsample_ratio=rot_params["subsample_ratio"],
        subsample_min=rot_params["subsample_min"],
        subsample_max=rot_params["subsample_max"],
        eps=eps,
        copies=rot_params["copies"],
    )  # output column of (n * d * d) flattened array of flattened matrices

    # Reshape the maps
    wects = unflatten_to_matrix(
        wects, num_heights * num_directions
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
#         return_dtype=pl.Array(pl.Float64, embedder.emb_dim),
#         is_elementwise=True,
#     )


def get_premapped_3D_triangle_wects(
    df: pl.LazyFrame | pl.DataFrame,
    vertices: str,
    triangles: str,
    normals: str,
    wect_params: dict,
    rot_params: dict,
    wname: str,
) -> pl.LazyFrame:
    """Compute the WECTs for the given mesh data and transofrmations.

    TODO: Verify the rest of this docstring.

    If an embedder is provided, the WECTs are embedded using the embedder.

    The needed input columns are:
        - vertices: A column with the vertices of the mesh, each a 2D array
        - triangles: A column with the triangles of the mesh, each a 2D array
        - normals: A column with the normals of the mesh, each a vector

    The output columns are:
        - wname: the name of the WECTs column, each a list of flattened arrays.

    A row in the output is uniquely identified using the id and wid columns.

    Args:
        df: A lazyframe or dataframe with the mesh data.
        vertices: The name of the vertices column.
        triangles: The name of the triangles column.
        normals: The name of the normals column.
        wect_params: The parameters for the WECT computation.
        rot_params: The parameters for the pre-alignment of the mesh.

    Returns:
        A lazyframe with the computed WECTs. The rows correspond to each WECT
        given by a transformation on an object.

    """
    eps = None
    if rot_params["heur_fix"]:
        eps = rot_params["eps"]

    vdim = 3  # TODO: unhardcode

    return df.lazy().with_columns(
        get_premapped_wect_3D_triangles_expr(
            vdim,
            vertices,
            triangles,
            normals,
            eps,
            rot_params,
            wect_params,
        ).alias(wname)
    )


def get_premapped_wects(
    df: pl.LazyFrame | pl.DataFrame,
    simplices: str,
    weights: str,
    provided_simplices: List[int],
    provided_weights: List[int],
    wect_params: dict,
    rot_params: dict,
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

    vdim = 3  # TODO: unhardcode

    return df.lazy().with_columns(
        get_premapped_wect_expr(
            vdim,
            simplices,
            weights,
            provided_simplices,
            provided_weights,
            rot_params,
            wect_params,
        ).alias(wname)
    )
