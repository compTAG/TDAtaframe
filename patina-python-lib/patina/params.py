"""Classes to hold query parameters."""

from typing import List, Optional


class ComplexInfo:
    """Info about a simplicial complex as stored in a polars dataframe."""

    def __init__(
        self, complex_name: str, vertices_dimension: int, provided_simplices: List[int]
    ) -> None:
        """Initialize a ComplexInfo.

        Args:
            complex_name: The name of the column pointing to a simplicial complex.
            vertices_dimension: The dimension of the stored vertices.
            provided_simplices: A list indicating the dimensions of stored simplices, in sorted order. This list must exclude 0.
        """
        self.simplices = complex_name
        self.vdim = vertices_dimension
        self.provided_simplices = provided_simplices


class WeightedComplexInfo(ComplexInfo):
    """Info about a weighted simplicial complex stored in a polars dataframe."""

    def __init__(
        self,
        complex_name: str,
        weights_name: str,
        vertices_dimension: int,
        provided_simplices: List[int],
        provided_weights: List[int],
    ) -> None:
        """Initialize a WeightedComplexInfo.

        Args:
            complex_name: The name of the column pointing to a simplicial complex.
            weights_name: The name of the column pointing to weights of a complex.
            vertices_dimension: The dimension of the stored vertices.
            provided_simplices: A list indicating the dimensions of stored simplices, in sorted order. This list must exclude 0.
            provided_weights: A list indicating the dimensions of stored weights, in sorted order. This list may contain 0.
        """
        super().__init__(complex_name, vertices_dimension, provided_simplices)
        self.weights = weights_name
        self.provided_weights = provided_weights


class MapArgs:
    """Info about how to map a simplicial complex."""

    def __init__(
        self,
        align_dimension: int,
        subsample_ratio: float,
        subsample_min: int,
        subsample_max: int,
        eps: Optional[float],
        copies: bool,
    ) -> None:
        """Initialize a MapArgs.

        Args:
            align_dimension: The dimension of simplices to use for computing the mapping.
            subsample_ratio: The ratio of points to use for computing a mapping
            subsample_min: Restricts the minimum number of points after subsampling.
            subsample_max: The maximum number of points to use for subsampling.
            eps: If given, a threshold for which to heuristically fix a map.
            copies: Whether to return all rotated + reflected copies of a map.
        """
        self.align_dimension = align_dimension
        self.subsample_ratio = subsample_ratio
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.eps = eps
        self.copies = copies


class WectArgs:
    """Parameters for computing the wect."""

    def __init__(self, num_directions: int, num_filt_steps: int) -> None:
        """Initialize a WectArgs.

        Args:
            num_directions: The number of directions used in computing the wect.
            num_filt_steps: The number of filtration steps used in computing the wect.
        """
        self.directions = num_directions
        self.steps = num_filt_steps
