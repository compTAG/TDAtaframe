"""Implement queries for polars dataframes to be used in shape-matching.

Specifically, these queries support the following pipeline:
    - Compute pre-alignment of mesh
    - Compute WECT on the pre-aligned mesh
    - Compute an embedding/post-processing of the WECT.

This whole pipeline can be called using compute_db_entry().
"""

import polars as pl
from ..utils import flatten_matrix, unflatten_to_matrix
from .register import barycenters, maps_svd


def get_barycenters(
    df: pl.LazyFrame,
    v: str,
    t: str,
    b: str,
) -> pl.LazyFrame:
    """Compute the barycenters of all meshes.

    Args:
        df: A lazyframe or dataframe with the vertices and triangles columns.
        v: name of the vertex column of type list[array[_,3]]
        nv: number of vertices per point cloud.
        nt: number of triangles per point cloud.
        t: name of the triangle column of type list[array[uint32,3]]. These
        b: name of the new barycenters column.
        id: name of the unique ID column.

    Return:
        A pl.LazyFrame including a column with the barycenters.
    """
    return df.lazy().with_columns(
        unflatten_to_matrix(
            barycenters(
                flatten_matrix(pl.col(v)),
                flatten_matrix(pl.col(t)),
            ),
            3,  # TODO: unhardcode stride using schema
        ).alias(b)
    )


def get_maps_svd_expr(vdim, vcol, scol, ncol, eps, rot_params):  # noqa
    maps = maps_svd(
        flatten_matrix(pl.col(vcol)),
        flatten_matrix(pl.col(scol)),
        flatten_matrix(pl.col(ncol)),
        subsample_ratio=rot_params["subsample_ratio"],
        subsample_min=rot_params["subsample_min"],
        subsample_max=rot_params["subsample_max"],
        eps=eps,
        copies=rot_params["copies"],
    )  # output column of (n * d * d) flattened array of flattened matrices

    # Reshape the maps
    maps = unflatten_to_matrix(
        maps, vdim * vdim
    )  # now (n)-list of d * d flattened matrices

    return maps


def get_maps_svd(
    df: pl.LazyFrame,
    rot_params: dict,
    txname: str,
) -> pl.LazyFrame:
    """Compute the mappings of each mesh in the dataframe, according to the given parameters.

    This maps the input shape so that it is axis aligned,
    where X is aligned with the largest principal component, Y the second, and
    Z the third. To fix possible rotations around these axes, specify heur_fix
    or copies in rot_params. See get_heur_fixes and get_copies for details.

    Args:
        df: A lazyframe or dataframe with vertices, normals, and
            triangles columns. There should be one object per row.
        rot_params: A dict specifying the behavior, refer to implementation
            for needed fields.
        txname: the name of the output column for the computed mappings.

    Returns:
        A lazyframe with the computed mappings.
    """

    eps = None
    if rot_params["heur_fix"]:
        eps = rot_params["eps"]

    vdim = 3  # TODO: unhardcode

    return df.with_columns(
        get_maps_svd_expr(
            vdim,
            "vertices",
            "triangles",
            "normals",
            eps,
            rot_params,
        ).alias(txname)
    )
