"""Classes to hold query parameters."""

from typing import List, Optional

from pydantic import BaseModel


class ComplexInfo(BaseModel):
    """Info about a simplicial complex as stored in a polars dataframe."""

    simplices: str  # The name of the column pointing to a simplicial complex
    vdim: int  # The dimension of the stored vertices
    provided_simplices: List[
        int
    ]  # A list indicating the dimensions of stored simplices, in sorted order. This list must exclude 0.


class WeightedComplexInfo(BaseModel):
    """Info about a weighted simplicial complex stored in a polars dataframe."""

    simplices: str  # The name of the column pointing to a simplicial complex
    weights: str  # The name of the column pointing to weights of a complex
    vdim: int  # The dimension of the stored vertices
    provided_simplices: List[
        int
    ]  # A list indicating the dimensions of stored simplices, in sorted order. This list must exclude 0.
    provided_weights: List[
        int
    ]  # A list indicating the dimensions of stored weights, in sorted order. This list may contain 0.


# convert to pydantic
class MapArgs(BaseModel):
    """Info about how to map a weighted simplicial complex."""

    align_dimension: int  # The dimension of simplices to use for computing the mapping.
    subsample_ratio: float  # The ratio of points to use for computing a mapping
    subsample_min: int  # Restricts the minimum number of points after subsampling.
    subsample_max: int  # The maximum number of points to use for subsampling.


class MapCopyArgs(BaseModel):
    """Info about how to map a weighted simplicial complex, when we want to see all possible maps given some confidence value."""

    align_dimension: int  # The dimension of simplices to use for computing the mapping.
    subsample_ratio: float  # The ratio of points to use for computing a mapping
    subsample_min: int  # Restricts the minimum number of points after subsampling.
    subsample_max: int  # The maximum number of points to use for subsampling.
    eps: Optional[
        float
    ]  # If given, a threshold for which to heuristically fix a map. This is the referred confidence value
    copies: bool  # Whether to return all rotated + reflected copies of a map.


class WectArgs(BaseModel):
    """Parameters for computing the wect."""

    directions: int  # The number of directions used in computing the wect.
    steps: int  # The number of filtration steps used in computing the wect.
