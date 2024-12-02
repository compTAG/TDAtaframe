import os
from typing import List

import numpy as np
import polars as pl
import networkx as nx

from tdataframe.loading import Loader


class GraphLoader(Loader):
    """Loader for graphs to use as simplicial complexes"""

    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        """Load graphs into a dataframe given vertices and edges.

        Returns a polars dataframe with the following fields:
            simplices - struct with edges and vertices
            ID - name

        Args:
            files: list of file names

        Returns:
            Dictionary with the fields as described above
        """
        d = {
            "simplices": {
                "vertices": [],
                "edges": [],
            },
            "ID": [],
        }
        for _, file in enumerate(files):
            vertices, edges, _ = self.parse(file)
            d["simplices"]["vertices"].append(vertices.tolist())
            d["simplices"]["edges"].append(edges.tolist())
            d["ID"].append(os.path.basename(file))

        return pl.DataFrame(d)


class GraphMlLoader:
    def parse(self, file: str) -> List[np.ndarray]:
        """Parse an embedded graph from a graphml file. return list of vertices, edges, and positions."""
        # Load the graph from the GraphML file
        G = nx.read_graphml(file)

        # Get the list of nodes (vertices)
        vertices = np.array(list(G.nodes()))

        # Get the list of edges
        edges = np.array(list(G.edges()))

        # Check if position data ('x' and 'y') is available for nodes
        positions = None
        node_attrs = G.nodes[next(iter(G.nodes()))]
        if "x" in node_attrs and "y" in node_attrs:
            positions = np.array([
                [float(G.nodes[node]["x"]), float(G.nodes[node]["y"])]
                for node in G.nodes()
            ])

        return [vertices, edges, positions]
