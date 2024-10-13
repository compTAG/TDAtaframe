import os
import pprint

import numpy as np
from tdataframe.ect import with_premapped_copy_wects, with_wects, with_premapped_wects
import polars as pl
from tdataframe.params import MapArgs, MapCopyArgs, EctArgs, WeightedComplexInfo
from tdataframe.alignment import with_barycenters

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA = {
    "ID": pl.String,
    "simplices": pl.Struct({
        "vertices": pl.List(pl.Float32),
        "triangles": pl.List(pl.UInt32),
    }),
    "weights": pl.Struct({
        "trinormals": pl.List(pl.Float32),
    }),
}


def test_overlap_triangle():
    # Define the vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 1
            [0.5, 1, 0],  # Vertex 2
        ],
        dtype=np.float32,
    )
    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 1, 2] for _ in range(2)],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
            1,
        ],
        dtype=np.float32,
    )

    degen_wect(vertices, faces, normals)


# def test_collapsed_triangle_to_origin_indexed(): # FIXME: FAILS
#     # Define the vertices
#     vertices = np.array(
#         [
#             [0, 0, 0],  # Vertex 0
#         ],
#         dtype=np.float32,
#     )
#
#     # Define the triangular faces using vertex indices
#     faces = np.array(
#         [[0, 0, 0]],
#         dtype=np.uint32,
#     )
#
#     normals = np.array(
#         [
#             1,
#         ],
#         dtype=np.float32,
#     )
#     degen_wect(vertices, faces, normals)
#
#
# def test_collapsed_triangle_to_origin_disjoint():  # FIXME: FAILS
#     # Define the vertices
#     vertices = np.array(
#         [
#             [0, 0, 0],  # Vertex 0
#             [0, 0, 0],  # Vertex 0
#             [0, 0, 0],  # Vertex 0
#         ],
#         dtype=np.float32,
#     )
#
#     # Define the triangular faces using vertex indices
#     faces = np.array(
#         [[0, 1, 2]],
#         dtype=np.uint32,
#     )
#
#     normals = np.array(
#         [
#             1,
#         ],
#         dtype=np.float32,
#     )
#     degen_wect(vertices, faces, normals)


def test_collapsed_triangle_to_origin_indexed_with_another_tri():
    # Define the vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 1
            [0.5, 1, 0],  # Vertex 2
        ],
        dtype=np.float32,
    )

    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 0, 0], [0, 1, 2]],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
            1,
        ],
        dtype=np.float32,
    )
    degen_wect(vertices, faces, normals)


def test_collapsed_triangle_2_to_origin_indexed():
    # Define the vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 1
            [0.5, 1, 0],  # Vertex 2
        ],
        dtype=np.float32,
    )

    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 0, 0], [0, 0, 0]],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
            1,
        ],
        dtype=np.float32,
    )
    degen_wect(vertices, faces, normals)


def test_collapsed_triangle_to_point_disjoint():
    # Define the vertices
    vertices = np.array(
        [
            [1, 1, 1],  # Vertex 0
            [1, 1, 1],  # Vertex 0
            [1, 1, 1],  # Vertex 0
        ],
        dtype=np.float32,
    )

    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 1, 2]],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
        ],
        dtype=np.float32,
    )
    degen_wect(vertices, faces, normals)


def test_collapsed_triangle_to_point_indexed():
    # Define the vertices
    vertices = np.array(
        [
            [1, 0, 0],  # Vertex 0
        ],
        dtype=np.float32,
    )

    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 0, 0]],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
        ],
        dtype=np.float32,
    )
    degen_wect(vertices, faces, normals)


def test_collapse_edge_to_origin():
    # Define the vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 0
        ],
        dtype=np.float32,
    )

    # Define the triangular faces using vertex indices
    faces = np.array(
        [[0, 0, 1]],
        dtype=np.uint32,
    )

    normals = np.array(
        [
            1,
        ],
        dtype=np.float32,
    )
    degen_wect(vertices, faces, normals)


def degen_wect(degen_vertices, degen_triangles, degen_normals) -> None:
    # vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    data = {
        "ID": ["degen"],
        "simplices": {
            "vertices": [
                degen_vertices.tolist(),
            ],
            "triangles": [
                degen_triangles.tolist(),
            ],
        },
        "weights": {
            "trinormals": [
                degen_normals.tolist(),
            ],
        },
    }
    df = pl.DataFrame(data)

    provided_weights = [2]

    wci = WeightedComplexInfo(
        simplices="simplices",
        weights="weights",
        provided_weights=provided_weights,
    )

    ea = EctArgs(directions=25, steps=20)

    print(df)

    wdf = with_wects(
        df.lazy(),
        wci,
        ea=ea,
        wname="wects",
    ).select("ID", "simplices", "weights", "wects")
    print(wdf.explain(streaming=True))
    wdf = wdf.collect()

    print(wdf)

    wects = wdf.to_dict()["wects"].to_numpy()
    for wect in wects:
        print(wect.reshape(25, 20))
