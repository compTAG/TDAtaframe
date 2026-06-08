from abc import ABC, abstractmethod
from typing import List

import numpy as np
import polars as pl


class Loader(ABC):
    """Abstract class for loading data from files."""

    @abstractmethod
    def parse(self, file: str) -> List[np.ndarray]:
        # `parse` is the file-format-specific layer; concrete loaders translate
        # external files into numpy arrays with one array per logical payload.
        """Parse a file and return the data as numpy arrays."""
        pass

    @abstractmethod
    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        # `load_complexes` is the DataFrame-construction layer; it shapes parsed
        # arrays into the struct/list columns expected by the Rust plugin.
        """Load simplices into a dataframe given a list of file formats and a loading function."""
        pass
