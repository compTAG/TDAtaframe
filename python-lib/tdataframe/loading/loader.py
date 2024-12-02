from abc import ABC, abstractmethod
from typing import List

import numpy as np
import polars as pl


class Loader(ABC):
    """Abstract class for loading data from files."""

    @abstractmethod
    def parse(self, file: str) -> List[np.ndarray]:
        """Parse a file and return the data as numpy arrays."""
        pass

    @abstractmethod
    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        """Load simplices into a dataframe given a list of file formats and a loading function."""
        pass
