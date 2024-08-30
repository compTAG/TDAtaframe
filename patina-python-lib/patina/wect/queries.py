"""Implement queries for polars dataframes to be used in shape-matching.

Specifically, these queries support the following pipeline:
    - Compute pre-alignment of mesh
    - Compute WECT on the pre-aligned mesh
    - Compute an embedding/post-processing of the WECT.

This whole pipeline can be called using compute_db_entry().
"""

import polars as pl
from ..utils import flatten_matrix, unflatten_to_matrix
from .register import pre_align_wect


def get_pred_mapped_wect_expr(vdim, vcol, tcol, ncol, eps, rot_params, wect_params):  # noqa
    num_heights = wect_params["steps"]
    num_directions = wect_params["directions"]
    wects = pre_align_wect(
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

    # res = (
    #     df.lazy()
    #     .select([id, v, t, n, nv, nt, txs])
    #     .with_columns(pl.int_ranges(pl.col(txs).list.len()).alias(wid))
    #     .with_row_index("i")
    #     .with_columns(pl.col("i").repeat_by(pl.col(txs).list.len()).alias("irep"))
    #     .cache()
    # )
    #
    # wects = res.select(
    #     _get_wects_expr(
    #         pl.col(v),
    #         pl.col(t),
    #         pl.col(n),
    #         pl.col(nv),
    #         pl.col(nt),
    #         pl.col(txs),
    #         pl.col("irep"),
    #         directions,
    #         steps,
    #     )
    #     .reshape((1, steps * directions))
    #     .alias(w),
    #     pl.col("irep").explode(),
    #     pl.col("wid").explode(),
    # )
    #
    # res = (
    #     res.select([id, txs, wid, "irep"])
    #     .explode(txs, "irep", wid)
    #     .with_columns(pl.col(txs).reshape((1, 9)))
    #     .join(wects, on=["irep", wid])
    # )
    #
    # if embedder is not None:
    #     output_name = w
    #     if emb_suffix is not None:
    #         output_name += emb_suffix
    #
    #     res = res.with_columns(
    #         _get_embedding_expr(pl.col(w), embedder).alias(output_name)
    #     )
    #     if output_name == w:
    #         res = res.select([id, txs, w, wid])
    #     else:
    #         res = res.select([id, txs, w, wid, output_name])
    # else:
    #     res = res.select([id, txs, w, wid])
    # return res


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


def get_premapped_wects(
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

    return df.with_columns(
        get_pred_mapped_wect_expr(
            vdim,
            vertices,
            triangles,
            normals,
            eps,
            rot_params,
            wect_params,
        ).alias(wname)
    )
