import numpy as np
import polars as pl
import os
import pprint


from patina.alignment import get_barycenters, get_maps_svd
from patina.wect import get_premapped_wects

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def build_octahedron(weighted: bool, x: float = 1.0, y: float = 1.0, z: float = 1.0):
    # Define the vertices
    vertices = np.array([
        [x, 0, 0],  # Vertex 0
        [-x, 0, 0],  # Vertex 1
        [0, y, 0],  # Vertex 2
        [0, -y, 0],  # Vertex 3
        [0, 0, z],  # Vertex 4
        [0, 0, -z],  # Vertex 5
    ])
    # Define the triangular faces using vertex indices
    faces = np.array([
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [0, 5, 2],
        [2, 5, 1],
        [1, 5, 3],
        [3, 5, 0],
    ])

    non_uniform = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    uniform = np.ones((8,), dtype=np.float32)

    weights = non_uniform if weighted else uniform
    return vertices, faces, weights


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
    vertices2, triangles2, normals2 = build_octahedron(True, 1.0, 1.0, 1.0)
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "octahedron2"],
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

    wdf = (
        get_premapped_wects(
            df.lazy(),
            vertices="vertices",
            triangles="triangles",
            normals="normals",
            wect_params=dict(directions=20, steps=25),
            rot_params=dict(
                heur_fix=True,
                eps=0.01,
                subsample_ratio=1.0,
                subsample_min=10,
                subsample_max=100,
                copies=False,
            ),
            wname="wects",
        )
        .select("ID", "vertices", "triangles", "wects")
        .collect()
    )
    pprint.pprint(wdf.to_numpy(), width=200)
    wects = wdf.to_dict()["wects"].to_numpy()
    pprint.pprint(wects, width=200)
