import os
from typing import List

import numpy as np
import polars as pl
import trimesh
import pywavefront

from tdataframe.params import WeightedComplexInfo
from tdataframe.loading import Loader


# Helper functions #
def interpolate_vertex_normals(
    normals: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Compute normal vectors for triangles by averaging vertex normals."""
    t_normals = []
    for triangle in triangles:
        # Retrieve the vertex normals of the face
        n1 = np.array(normals[triangle[0]])
        n2 = np.array(normals[triangle[1]])
        n3 = np.array(normals[triangle[2]])

        normal = (n1 + n2 + n3) / 3
        t_normals.append(normal)

    return np.asarray(t_normals)


def calculate_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute unit normal vectors from vertices and triangles.

    Assumes a right-hand rule ordering of triangle vertices.

    Args:
        vertices: list of vertex coordinates
        triangles: list of triangles as indices

    Returns:
        A list of unit normal vectors, one for each triangle.
    """
    normals = []
    for triangle in triangles:
        # Retrieve the vertices of the face
        p1 = np.array(vertices[triangle[0]])
        p2 = np.array(vertices[triangle[1]])
        p3 = np.array(vertices[triangle[2]])

        # compute vectors at p1
        v1 = p3 - p1
        v2 = p2 - p1
        normal = np.cross(v1, v2)

        # Normalize the normal (to length 1)
        normal = normal / np.linalg.norm(normal)

        normals.append(normal.tolist())

    return np.asarray(normals)


# General Mesh Loaders #


class MeshLoader(Loader):
    """A loader for surface meshes."""

    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        """Load meshes into a dataframe given a list of file formats and a loading function.

        Returns a polars dataframe with the following fields:
            simplices - struct with vertices and triangles
            ID - name

        Args:
            files: list of file names
            load_func: function to load the mesh, returns vertices, triangles, and normals

        Returns:
            Dictionary with the fields as described above
        """
        d = {
            "simplices": {
                "vertices": [],
                "triangles": [],
            },
            "ID": [],
        }
        for _, file in enumerate(files):
            vertices, triangles, _ = self.parse(file)
            d["simplices"]["vertices"].append(vertices.tolist())
            d["simplices"]["triangles"].append(triangles.tolist())
            d["ID"].append(os.path.basename(file))

        return pl.DataFrame(d)


class WeightedFaceMeshLoader(Loader):
    """A loader for surface meshes with weighted faces."""

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self.wci = WeightedComplexInfo(
            simplices="simplices", weights="weights", provided_weights=[2]
        )

    def get_wci(self) -> WeightedComplexInfo:
        """Return the WeightedComplexInfo representing the complexes as loaded into a dataframe."""
        return self.wci

    def load_complexes(self, files: List[str]) -> pl.DataFrame:
        """Load weighted meshes into a dataframe given a list of file formats and a loading function.

        Returns a polars dataframe with the following fields:
            simplices - struct with vertices and triangles
            weights - weights for each triangle
            ID - name

        Args:
            files: list of file names
            load_func: function to load the mesh, returns vertices, triangles, and normals

        Returns:
            Dictionary with the fields as described above
        """
        d = {
            "simplices": {
                "vertices": [],
                "triangles": [],
            },
            "weights": {
                "normals": [],
            },
            "ID": [],
        }
        for _, file in enumerate(files):
            vertices, triangles, normals = self.parse(file)
            d["simplices"]["vertices"].append(vertices.tolist())
            d["simplices"]["triangles"].append(triangles.tolist())
            d["weights"]["normals"].append(normals.tolist())
            d["ID"].append(os.path.basename(file))

        return pl.DataFrame(d)


class WeightedObjLoader(WeightedFaceMeshLoader):
    """Loader for OBJ files."""

    def __init__(self, magnitude: bool = True) -> None:
        """Initialize the loader.

        Args:
            magnitude: use the magnitude of interpolated normal vectors for faces.
        """
        super().__init__()
        self.magnitude = magnitude

    def parse(self, file: str) -> List[np.ndarray]:
        """Load an OBJ file.

        Return list of vertex coordinates, list of
        triangles as indices, and a list of normal vectors.

        Args:
            file: name of file to load, must be OBJ format.

        Returns:
            Tuple of vertices, triangles, and normals.
        """
        scene = pywavefront.Wavefront(file, collect_faces=True, create_materials=True)
        vertices = np.asarray(scene.vertices)
        triangles = np.asarray(scene.mesh_list[0].faces)
        normals = np.asarray(scene.parser.normals)

        if len(vertices) == len(normals):
            normals = interpolate_vertex_normals(normals, triangles)
            if self.magnitude:
                normals = np.abs(normals)
        else:
            if self.magnitude:
                normals = np.ones(
                    len(triangles), dtype=np.float32
                )  # default to unit magnitude
            else:
                normals = calculate_normals(vertices, triangles)  # calc unit norms

        return [vertices, triangles, normals]


class StlLoader(MeshLoader):
    """Loader for STL files."""

    def parse(self, file: str) -> List[np.ndarray]:
        """Load an STL file.

        Returns list of vertex coordinates, list of
        triangles as indices, and ones for face normals.
        """
        mesh = trimesh.load_mesh(file)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        triangles = np.array(mesh.faces, dtype=np.uint32)
        return [vertices, triangles, np.ones(len(triangles), dtype=np.float32)]
