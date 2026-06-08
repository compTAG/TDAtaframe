import polars as pl


def flatten_matrix(expr: pl.Expr) -> pl.Expr:
    """Flatten column of 2D arrays to just a list[float].

    If a higher dimensional array is passed, the function will flatten the
    top dimension of the array. For example, if the input is a column of 3D
    arrays, the function will flatten the top dimension to a list of 2D arrays.

    Args:
        expr: The column to flatten.

    Returns:
        A column of flattened arrays.
    """
    # The Rust plugin returns flat lists for compatibility with Polars plugin
    # output types; this helper makes that reshaping explicit in Python code.
    return expr.list.eval(pl.element().flatten())


def unflatten_to_matrix(expr: pl.Expr, stride: int) -> pl.Expr:
    """Unflatten a column of 1D arrays to a column of 2D arrays.

    Args:
        expr: The column to unflatten.
        stride: The stride of the unflattened arrays. This becomes the width
            of the 2D arrays.

    Returns:
        A column of 2D arrays, with the stride as the width.
    """
    # `stride` is the width of the inner matrix. The outer length is inferred
    # from the flat list length for each row independently.
    return expr.list.eval(pl.element().reshape((-1, stride)))


def top_dim_count(expr: pl.Expr) -> pl.Expr:
    """Get the length of the top dimension of a multidimensional list."""
    # This is mainly used after copy-generating transforms/WECTs, where one
    # input object can expand into multiple candidate outputs.
    return expr.list.eval(pl.element().len()).explode()


def concat_id(
    df: pl.LazyFrame, ending: str, id: str = "ID", separator: str = "-"
) -> pl.LazyFrame:
    """Concatenate the ID column with the entry from the ending column.

    Args:
        df: A lazyframe with the ID and ending columns.
        ending: The column to concatenate with the ID.
        id: The name of the ID column.
        separator: The separator to use between the ID and ending.

    Returns:
        A lazyframe with the ID column concatenated with the ending column.
    """
    # The alignment pipelines often fan one source object out into several rows;
    # appending a suffix keeps those derived IDs traceable.
    return df.with_columns(
        pl.concat_str([pl.col(id), pl.col(ending)], separator=separator).alias(id)
    )


def l2norm(col: pl.Expr) -> pl.Expr:
    """Get the l2 norm of each array in the column."""
    return (
        # `arr` expressions do not expose a direct vector norm in the shape used
        # here, so we go through list evaluation to stay lazy and columnar.
        col.arr.to_list()
        .list.eval(pl.element().pow(2).sum().sqrt())
        .explode()  # turn single elem lists to elems
    )
