import numpy as np
import pywavefront
from typing import List, Tuple


def load_obj(
    filename: str, magnitude: bool = True
) -> Tuple[List[List[float]], List[List[int]], List[float]]:
    """Load an OBJ file.

    Return list of vertex coordinates, list of
    triangles as indices, and a list of normal vectors.

    Args:
        filename: name of file to load, must be OBJ format.
        magnitude: use the magnitude of interpolated normal vectors for faces.

    Returns:
        Tuple of vertices, triangles, and normals.
    """
    scene = pywavefront.Wavefront(filename, collect_faces=True, create_materials=True)
    vertices = scene.vertices
    triangles = scene.mesh_list[0].faces
    normals = scene.parser.normals

    if len(vertices) == len(normals):
        normals = interpolate_vertex_normals(normals, triangles)
        if magnitude:
            normals = np.abs(normals)
    else:
        if magnitude:
            normals = np.ones(
                len(triangles), dtype=np.float32
            )  # default to unit magnitude
        else:
            normals = calculate_normals(vertices, triangles)  # calc unit norms

    return vertices, triangles, normals


def calculate_normals(vertices: List[List[float], triangles: List[List[int]]) -> NDFloat:
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


def interpolate_vertex_normals(
    normals: List[float], triangles: List[List[int]]
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
