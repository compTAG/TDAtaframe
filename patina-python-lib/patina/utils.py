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
    return expr.list.eval(pl.element().reshape((-1, stride)))


def top_dim_count(expr: pl.Expr) -> pl.Expr:
    """Get the length of the top dimension of a multidimensional list."""
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
    return df.with_columns(
        pl.concat_str([pl.col(id), pl.col(ending)], separator=separator).alias(id)
    )


def l2norm(col: pl.Expr) -> pl.Expr:
    """Get the l2 norm of each array in the column."""
    return (
        col.arr.to_list()
        .list.eval(pl.element().pow(2).sum().sqrt())
        .explode()  # turn single elem lists to elems
    )
