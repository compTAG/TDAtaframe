from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl

from ..ect import with_ects, with_wects
from ..params import EctArgs, WeightedComplexInfo

DescriptorName = Literal["ect", "wect"]


def build_descriptor_entries(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    descriptor: DescriptorName,
    ect_args: EctArgs,
    descriptor_column: str = "descriptor",
    id_column: str = "ID",
    keep_columns: Sequence[str] | None = None,
    simplex_column: str = "simplices",
    wci: WeightedComplexInfo | None = None,
) -> pl.LazyFrame:
    """Build a retrieval-ready descriptor table from loaded objects.

    This is the user-facing orchestration layer between raw complex tables from
    ``tdataframe.loading`` and search helpers from ``tdataframe.retrieval``.

    Args:
        df: Loaded objects in the canonical dataframe schema.
        descriptor: Which descriptor family to compute.
        ect_args: Shared ECT/WECT discretization settings.
        descriptor_column: Name of the output descriptor vector column.
        id_column: Name of the stable identifier column to preserve.
        keep_columns: Extra columns to retain in the returned lookup table.
        simplex_column: Name of the simplices struct column for ECT.
        wci: Weighted complex metadata required for WECT.

    Returns:
        A lazyframe containing the identifier, any requested passthrough
        columns, and one computed descriptor column.
    """
    extra_columns = list(keep_columns or [])
    columns = _output_columns(
        id_column=id_column,
        descriptor_column=descriptor_column,
        keep_columns=extra_columns,
    )

    if descriptor == "ect":
        built = with_ects(
            df.lazy(),
            simplex_column=simplex_column,
            ea=ect_args,
            ename=descriptor_column,
        )
    elif descriptor == "wect":
        if wci is None:
            raise ValueError("wci is required when descriptor='wect'")
        built = with_wects(
            df.lazy(),
            wci=wci,
            ea=ect_args,
            wname=descriptor_column,
        )
    else:
        raise ValueError(f"Unsupported descriptor {descriptor!r}")

    return built.select(*columns)


def _output_columns(
    *,
    id_column: str,
    descriptor_column: str,
    keep_columns: Sequence[str],
) -> list[str]:
    columns: list[str] = []
    for name in [id_column, *keep_columns, descriptor_column]:
        if name not in columns:
            columns.append(name)
    return columns
