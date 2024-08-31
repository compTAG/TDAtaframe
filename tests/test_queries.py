import numpy as np
import polars as pl
import os
import pprint


from patina.alignment import get_barycenters, get_maps_svd
from patina.wect import get_premapped_wects

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


# TETRAHEDRON_VERTICES = np.array([
#     [0, 0, 1],
#     [2 * np.sqrt(2) / 3, 0, -1 / 3],
#     [-np.sqrt(2) / 3, np.sqrt(2), -1 / 3],
#     [-np.sqrt(2) / 3, -np.sqrt(2), -1 / 3],
# ])
#
# TETRAHEDRON_EDGES = np.array([
#     [0, 1],
#     [0, 2],
#     [0, 3],
#     [1, 2],
#     [1, 3],
#     [2, 3],
# ])
#
# TETRAHEDRON_TRIANGLES = np.array([
#     [0, 1, 2],
#     [0, 2, 3],
#     [0, 3, 1],
#     [1, 2, 3],
# ])


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


def test_bary() -> None:
    # build a simple bubble as an octahedron

    vertices, triangles, _ = build_octahedron(True, 2.0, 1.0, 3.0)
    df = pl.DataFrame(
        {
            "ID": ["octahedron"],
            "vertices": [vertices],
            "triangles": [triangles],
        },
        schema={
            "ID": pl.String,
            "vertices": pl.List(pl.Array(pl.Float32, 3)),
            "triangles": pl.List(pl.Array(pl.UInt32, 3)),
        },
    )

    target_bary = np.array([
        [0.6666667, 0.33333334, 1.0],
        [-0.6666667, 0.33333334, 1.0],
        [-0.6666667, -0.33333334, 1.0],
        [0.6666667, -0.33333334, 1.0],
        [0.6666667, 0.33333334, -1.0],
        [-0.6666667, 0.33333334, -1.0],
        [-0.6666667, -0.33333334, -1.0],
        [0.6666667, -0.33333334, -1.0],
    ])

    bdf = (
        get_barycenters(
            df.lazy(),
            v="vertices",
            t="triangles",
            b="barycenters",
        )
        .select("ID", "vertices", "triangles", "barycenters")
        .collect()
    )
    pprint.pprint(bdf.to_numpy(), width=200)
    barycenters = bdf.to_dict()["barycenters"].to_numpy()
    assert np.allclose(barycenters[0], target_bary)


def test_wect():
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    vertices2, _, triangles2, normals2 = build_tetrahedron()
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "tetrahedron"],
            "vertices": [vertices, vertices2],
            "triangles": [triangles, triangles2],
            "normals": [normals, normals2],
        },
        schema={
            "ID": pl.String,
            "vertices": pl.List(pl.Array(pl.Float32, 3)),
            "triangles": pl.List(pl.Array(pl.UInt32, 3)),
            "normals": pl.List(pl.Float32),
        },
    )
    print(df)

    wdf = get_premapped_wects(
        df.lazy(),
        vertices="vertices",
        triangles="triangles",
        normals="normals",
        wect_params=dict(directions=25, steps=20),
        rot_params=dict(
            heur_fix=True,
            eps=0.01,
            subsample_ratio=1.0,
            subsample_min=10,
            subsample_max=100,
            copies=False,
        ),
        wname="wects",
    ).select("ID", "vertices", "triangles", "wects")
    print(wdf.explain(streaming=True))
    wdf = wdf.collect()

    wects = wdf.to_dict()["wects"].to_numpy()
    for wect in wects:
        wect = wect.reshape(-1, 25, 20)
        print(wect)
