"""Implement queries for polars dataframes to be used in shape-matching.

Specifically, these queries support the following pipeline:
    - Compute pre-alignment of mesh
    - Compute WECT on the pre-aligned mesh
    - Compute an embedding/post-processing of the WECT.

This whole pipeline can be called using compute_db_entry().
"""

import polars as pl
from ..utils import flatten_matrix, unflatten_to_matrix
from .register import barycenters as _barycenters, maps_svd as _maps_svd
from ..params import MapArgs


def with_barycenters(
    df: pl.LazyFrame,
    v: str,
    t: str,
    vdim: int,
    sdim: int,
    b: str,
    flat_in=True,
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
    if not flat_in:
        return df.lazy().with_columns(
            _barycenters(
                flatten_matrix(pl.col(v)),
                flatten_matrix(pl.col(t)),
                embedded_dimension=vdim,
                simplex_dimension=sdim,
            ).alias(b),
        )
    return df.lazy().with_columns(
        _barycenters(
            pl.col(v),
            pl.col(t),
            embedded_dimension=vdim,
            simplex_dimension=sdim,
        ).alias(b),
    )


def maps_svd(
    v: str, s: str, w: str, vdim: int, ma: MapArgs, flat_in: bool = True
) -> pl.Expr:
    """Compute the maps for the given vertices, normals, and triangles.

    Args:
        v: The name of the column containing the vertices.
        s: The name of the column containing the simplices.
        w: The name of the column containing the weights.
        vdim: The dimension of the vertices.
        ma: The arguments for the map computation.
        flat_in: if the inputs are already flattened.
    """
    if not flat_in:
        maps = _maps_svd(
            flatten_matrix(pl.col(v)),
            flatten_matrix(pl.col(s)),
            flatten_matrix(pl.col(w)),
            embedded_dimension=vdim,
            simplex_dimension=ma.align_dimension,
            subsample_ratio=ma.subsample_ratio,
            subsample_min=ma.subsample_min,
            subsample_max=ma.subsample_max,
            eps=ma.eps,
            copies=ma.copies,
        )  # output column of (n * d * d) flattened array of flattened matrices
    else:
        maps = _maps_svd(
            pl.col(v),
            pl.col(s),
            pl.col(w),
            embedded_dimension=vdim,
            simplex_dimension=ma.align_dimension,
            subsample_ratio=ma.subsample_ratio,
            subsample_min=ma.subsample_min,
            subsample_max=ma.subsample_max,
            eps=ma.eps,
            copies=ma.copies,
        )

    # Reshape the maps
    maps = unflatten_to_matrix(
        maps, vdim * vdim
    )  # now (n)-list of d * d flattened matrices

    return maps


def with_maps_svd(
    df: pl.LazyFrame,
    vertices: str,
    simplices: str,
    weights: str,
    vdim: int,
    ma: MapArgs,
    txname: str,
) -> pl.LazyFrame:
    """Compute the mappings of each mesh in the dataframe, according to the given parameters.

    Args:
        df: The dataframe containing the meshes.
        vertices: the name of the column containing the vertices, of type list[array[_,3]].
        txname: the name of the output column for the computed mappings.

    Returns:
        A lazyframe with the computed mappings.
    """

    return df.with_columns(
        maps_svd(vertices, simplices, weights, vdim, ma).alias(txname)
    )
