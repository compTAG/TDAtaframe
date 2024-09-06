"""Implement queries for polars dataframes to be used in shape-matching.

Specifically, these queries support the following pipeline:
    - Compute pre-alignment of mesh
    - Compute WECT on the pre-aligned mesh
    - Compute an embedding/post-processing of the WECT.

This whole pipeline can be called using compute_db_entry().
"""

import polars as pl

from ..utils import flatten_matrix, unflatten_to_matrix
from ..params import MapArgs, MapCopyArgs
from .register import (
    maps_svd_copies as _maps_svd_copies,
    barycenters as _barycenters,
    map_svd as _map_svd,
)


def with_barycenters(
    df: pl.LazyFrame,
    v: str,
    t: str,
    vdim: int,
    sdim: int,
    b: str,
    flat_in: bool = True,
) -> pl.LazyFrame:
    """Compute the barycenters of all meshes.

    Args:
        df: A lazyframe or dataframe with the vertices and triangles columns.
        v: name of the vertex column of type list[array[_,3]]
        t: name of the triangle column of type list[array[uint32,3]]. These
        vdim: dimension of the vertices.
        sdim: dimension of the simplices.
        b: name of the new barycenters column.
        flat_in: if the inputs are already flattened.

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


def map_svd(
    v: str, s: str, w: str, vdim: int, ma: MapArgs, flat_in: bool = True
) -> pl.Expr:
    """Compute the maps for the given vertices, normals, and triangles.

    For each mesh, the map computes the Vt from the SVD of the unweighted simplex barycenters.
    Then, the weighted centroid is computed, and a TX is computed which points the weighted centroid towards
    an all-positive n-drant. The TX is then applied to the Vt to get the final map.

    Args:
        v: The name of the column containing the vertices.
        s: The name of the column containing the simplices.
        w: The name of the column containing the weights.
        vdim: The dimension of the vertices.
        ma: The arguments for the map computation.
        flat_in: if the inputs are already flattened.
    """
    if not flat_in:
        map = _map_svd(
            flatten_matrix(pl.col(v)),
            flatten_matrix(pl.col(s)),
            flatten_matrix(pl.col(w)),
            embedded_dimension=vdim,
            simplex_dimension=ma.align_dimension,
            subsample_ratio=ma.subsample_ratio,
            subsample_min=ma.subsample_min,
            subsample_max=ma.subsample_max,
        )
    else:
        map = _map_svd(
            pl.col(v),
            pl.col(s),
            pl.col(w),
            embedded_dimension=vdim,
            simplex_dimension=ma.align_dimension,
            subsample_ratio=ma.subsample_ratio,
            subsample_min=ma.subsample_min,
            subsample_max=ma.subsample_max,
        )

    return map


def with_map_svd(
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
        simplices: the name of the column containing the simplices, of type list[array[uint32,3]].
        weights: the name of the column containing the weights, of type list[array[_,3]].
        vdim: the dimension of the vertices.
        ma: the arguments for the mapping computation.
        txname: the name of the output column for the computed mappings.

    Returns:
        A lazyframe with the computed mappings.
    """
    return df.with_columns(
        map_svd(vertices, simplices, weights, vdim, ma).alias(txname)
    )


def maps_svd_copies(
    v: str, s: str, w: str, vdim: int, ma: MapCopyArgs, flat_in: bool = True
) -> pl.Expr:
    """Compute the maps for the given vertices, normals, and triangles.

    If ma.eps is given, a single map is returned if the weighted centroid is
    below the threshold. Otherwise, multiple maps are returned.

    If ma.copies is True, the output is all possible copies.

    If neither eps nor copies are given, a single map is returned, Vt.

    Args:
        v: The name of the column containing the vertices.
        s: The name of the column containing the simplices.
        w: The name of the column containing the weights.
        vdim: The dimension of the vertices.
        ma: The arguments for the map computation.
        flat_in: if the inputs are already flattened.
    """
    if not flat_in:
        maps = _maps_svd_copies(
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
        maps = _maps_svd_copies(
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


def with_maps_svd_copies(
    df: pl.LazyFrame,
    vertices: str,
    simplices: str,
    weights: str,
    vdim: int,
    ma: MapCopyArgs,
    txname: str,
) -> pl.LazyFrame:
    """Compute the mappings of each mesh in the dataframe, according to the given parameters.

    Args:
        df: The dataframe containing the meshes.
        vertices: the name of the column containing the vertices, of type list[array[_,3]].
        simplices: the name of the column containing the simplices, of type list[array[uint32,3]].
        weights: the name of the column containing the weights, of type list[array[_,3]].
        vdim: the dimension of the vertices.
        ma: the arguments for the mapping computation.
        txname: the name of the output column for the computed mappings.

    Returns:
        A lazyframe with the computed mappings.
    """
    return df.with_columns(
        maps_svd_copies(vertices, simplices, weights, vdim, ma).alias(txname)
    )
