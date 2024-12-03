import os
from typing import List

import numpy as np
import polars as pl
from shapely import wkt
import networkx as nx

from tdataframe.loading import Loader


class GraphLoader(Loader):
    """Loader for graphs to use as simplicial complexes."""

    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        """Load graphs into a dataframe given vertices, edges, and polylines.

        Returns a polars dataframe with the following fields:
            simplices - struct with edges, vertices, and polylines
            ID - name

        Args:
            files: list of file names

        Returns:
            Polars DataFrame with the fields as described above
        """
        d = {
            "simplices": {
                "vertices": [],
                "edges": [],
            },
            "ID": [],
        }
        for _, file in enumerate(files):
            vertices, edges = self.parse(file)
            d["simplices"]["vertices"].append(vertices.tolist())
            d["simplices"]["edges"].append(edges.tolist())
            d["ID"].append(os.path.basename(file))

        return pl.DataFrame(d)


class GraphMlLoader(GraphLoader):
    """Loader for GraphML files to represent embedded graphs as simplicial complexes."""

    def parse(self, file: str) -> List[np.ndarray]:
        """Parse a graph from a GraphML file.

        The graph parsed is parsed as a set of nodes and
        edges (as polylines). To represent the graph as a
        simplicial complex, for each segment of each polyline edge
        we add each segment's endpoints to our vertex set and add the
        segment to our edge set.

        This loading does not consider the case where
        two polylines from the input graph cross eachother.

        Args:
            file: path to the GraphML file



        Return vertices, edges, and polylines.
        """
        # Load the graph from the GraphML file
        g = nx.read_graphml(file)

        # Get the list of nodes (vertices)
        nodes = list(g.nodes())
        node_indices = {v: i for i, v in enumerate(nodes)}

        # Check if position data ('x' and 'y') is available for nodes
        node_attrs = g.nodes[next(iter(g.nodes()))]
        if not ("x" in node_attrs and "y" in node_attrs):
            raise ValueError("Node positions ('x' and 'y') are required in the graph.")

        # Build an array of vertex positions, in the same order as 'nodes'
        vertices = [
            [float(g.nodes[node]["x"]), float(g.nodes[node]["y"])] for node in nodes
        ]

        # original_vertices_amount = len(vertices)

        # Get the list of edges and their polylines
        # edges = []
        polylines = []
        for u, v, data in g.edges(data=True):
            # edges.append((node_indices[u], node_indices[v]))
            if "geometry" in data:
                # Extract the geometry string and convert it to a LineString
                line_str = data["geometry"]
                line = list(wkt.loads(line_str).coords)
                for i in range(len(line) - 1):
                    _v0, v1 = line[i], line[i + 1]
                    if i == 0:  # v0 is first point
                        # Add second vertex to the list of vertices
                        vertices.append(list(v1))
                        polylines.append([node_indices[u], len(vertices) - 1])

                    elif i == len(line) - 2:  # v0 is Sscond to last point
                        # v0 most recently added already to the list of vertices
                        polylines.append([len(vertices) - 1, node_indices[v]])

                    else:  # Middle points
                        # Add each vertex to the list of vertices
                        # v0 is already added, so add v1
                        vertices.append(list(v1))
                        polylines.append([len(vertices) - 2, len(vertices) - 1])
            else:
                # If no geometry, use straight line between node positions
                polylines.append([
                    node_indices[u],
                    node_indices[v],
                ])

        vertices = np.array(vertices, dtype=np.float32)
        polylines = np.array(
            polylines,
            dtype=np.int32,
        )
        print(vertices)
        print(polylines)
        return [vertices, polylines]
