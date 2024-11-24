import polars as pl
import numpy as np


def build_tetrahedron():
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [2.0 * np.sqrt(2) / 3, 0.0, -1.0 / 3],
            [-np.sqrt(2) / 3, np.sqrt(2), -1.0 / 3],
            [-np.sqrt(2) / 3, -np.sqrt(2), -1.0 / 3],
        ],
        dtype=np.float32,
    )

    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=np.uint32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 2, 3],
        ],
        dtype=np.uint32,
    )

    weights = np.ones((4,), dtype=np.float32)
    return vertices, edges, faces, weights


def build_octahedron(weighted: bool, x: float = 1.0, y: float = 1.0, z: float = 1.0):
    # Define the vertices
    vertices = np.array(
        [
            [x, 0, 0],  # Vertex 0
            [-x, 0, 0],  # Vertex 1
            [0, y, 0],  # Vertex 2
            [0, -y, 0],  # Vertex 3
            [0, 0, z],  # Vertex 4
            [0, 0, -z],  # Vertex 5
        ],
        dtype=np.float32,
    )
    # Define the triangular faces using vertex indices
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [0, 5, 2],
            [2, 5, 1],
            [1, 5, 3],
            [3, 5, 0],
        ],
        dtype=np.uint32,
    )

    non_uniform = np.array(
        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    uniform = np.ones((8,), dtype=np.float32)

    weights = non_uniform if weighted else uniform
    return vertices, faces, weights


def test_loading_as_lists():
    # vertices is a 2D array of shape (6, 3)
    # triangles is a 2D array of shape (8, 3)
    # normals is a 1D array of shape (8,)
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    # vertices2 is a 2D array of shape (4, 3)
    # triangles2 is a 2D array of shape (4, 3)
    # normals2 is a 1D array of shape (4,)
    vertices2, _, triangles2, normals2 = build_tetrahedron()
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "tetrahedron"],
            "simplices": {
                "vertices": [vertices.tolist(), vertices2.tolist()],
                "triangles": [triangles.tolist(), triangles2.tolist()],
            },
            "weights": {
                "trinormals": [normals.tolist(), normals2.tolist()],
            },
        },
    )
    print(df)
    print(df.schema)


def test_loading_as_arrays():
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    vertices2, _, triangles2, normals2 = build_tetrahedron()
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "tetrahedron"],
            "simplices": {
                "vertices": [vertices, vertices2],
                "triangles": [triangles, triangles2],
            },
            "weights": {
                "trinormals": [normals, normals2],
            },
        },
    )
    print(df)
    print(df.schema)


def test_loading_as_lists_and_cast():
    # vertices is a 2D array of shape (6, 3)
    # triangles is a 2D array of shape (8, 3)
    # normals is a 1D array of shape (8,)
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    # vertices2 is a 2D array of shape (4, 3)
    # triangles2 is a 2D array of shape (4, 3)
    # normals2 is a 1D array of shape (4,)
    vertices2, _, triangles2, normals2 = build_tetrahedron()
    d = {
        "ID": pl.String,
        "simplices": pl.Struct({
            "vertices": pl.List(pl.List(pl.Float32)),
            "triangles": pl.List(pl.List(pl.Int64)),
        }),
    }
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "tetrahedron"],
            "simplices": {
                "vertices": [vertices.tolist(), vertices2.tolist()],
                "triangles": [triangles.tolist(), triangles2.tolist()],
            },
            "weights": {
                "trinormals": [normals.tolist(), normals2.tolist()],
            },
        },
    ).cast(d)

    print(df)
    print(df.schema)


if __name__ == "__main__":
    test_loading_as_lists_and_cast()
