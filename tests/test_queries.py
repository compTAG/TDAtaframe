import os
import pprint

import numpy as np
import polars as pl
from tdataframe.ect import (
    with_ects,
    # with_premapped_copy_ects,
    # with_premapped_ects,
    with_wects,
    with_premapped_wects,
    with_premapped_copy_wects,
)
from tdataframe.params import (
    EctArgs,
    MapArgs,
    MapCopyArgs,
    WeightedComplexInfo,
)
from tdataframe.alignment import with_barycenters

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
            "vertices": [vertices.tolist()],
            "triangles": [triangles.tolist()],
        },
        schema={
            "ID": pl.String,
            "vertices": pl.List(pl.List(pl.Float32)),
            "triangles": pl.List(pl.List(pl.UInt32)),
        },
    )

    print(df)

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
        with_barycenters(
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
    assert np.allclose(barycenters[0].reshape(-1, 3), target_bary)


def test_wect() -> None:
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
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
    print(wdf.explain(engine=True))
    wdf = wdf.collect()

    print(wdf)

    wects = wdf.to_dict()["wects"].to_numpy()
    for wect in wects:
        print(wect.reshape(25, 20))


# TODO: Add assertions for WECT queries
def test_premapped_wect() -> None:
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
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

    provided_weights = [2]

    wci = WeightedComplexInfo(
        simplices="simplices",
        weights="weights",
        provided_weights=provided_weights,
    )

    ea = EctArgs(directions=25, steps=20)
    ma = MapArgs(
        subsample_ratio=1.0,
        subsample_min=10,
        subsample_max=100,
        align_dimension=2,
    )

    print(df)

    wdf = with_premapped_wects(
        df.lazy(),
        wci,
        ma=ma,
        ea=ea,
        wname="wects",
    ).select("ID", "simplices", "weights", "wects")
    print(wdf.explain(engine=True))
    wdf = wdf.collect()

    wect = wdf.to_dict()["wects"].to_numpy()
    print(wect)


def test_premapped_copy_wect() -> None:
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
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

    provided_weights = [2]

    wci = WeightedComplexInfo(
        simplices="simplices",
        weights="weights",
        provided_weights=provided_weights,
    )

    ea = EctArgs(directions=25, steps=20)
    ma = MapCopyArgs(
        subsample_ratio=1.0,
        subsample_min=10,
        subsample_max=100,
        eps=0.01,
        copies=False,
        align_dimension=2,
    )

    print(df)

    wdf = with_premapped_copy_wects(
        df.lazy(),
        wci,
        ma=ma,
        ea=ea,
        wname="wects",
    ).select("ID", "simplices", "weights", "wects")
    print(wdf.explain(engine=True))
    wdf = wdf.collect()

    wects = wdf.to_dict()["wects"].to_numpy()
    for wect in wects:
        wect = wect.reshape(-1, 25, 20)
        print(wect)


def test_ect() -> None:
    vertices, triangles, _normals = build_octahedron(False, 2.0, 1.0, 3.0)
    vertices2, _, triangles2, _normals2 = build_tetrahedron()
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "tetrahedron"],
            "simplices": {
                "vertices": [vertices.tolist(), vertices2.tolist()],
                "triangles": [triangles.tolist(), triangles2.tolist()],
            },
        },
    )
    ea = EctArgs(directions=25, steps=20)

    print(df)
    print(df.schema)

    edf = with_ects(
        df.lazy(),
        "simplices",
        ea=ea,
        ename="ects",
    ).select("ID", "simplices", "ects")
    print(edf.explain(engine=True))
    edf = edf.collect()

    ects = edf.to_dict()["ects"].to_numpy()
    for ect in ects:
        print(ect.reshape(25, 20))


def test_ect32() -> None:
    vertices, triangles, _normals = build_octahedron(False, 2.0, 1.0, 3.0)
    vertices2, _, triangles2, _normals2 = build_tetrahedron()
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
        },
    ).cast(d)

    ea = EctArgs(directions=25, steps=20)

    print(df)
    print(df.schema)

    edf = with_ects(
        df.lazy(),
        "simplices",
        ea=ea,
        ename="ects",
    ).select("ID", "simplices", "ects")
    print(edf.explain(engine=True))
    edf = edf.collect()

    ects = edf.to_dict()["ects"].to_numpy()
    for ect in ects:
        print(ect.reshape(25, 20))


# def test_ect() -> None:
#     vertices, triangles, normals = build_octahedron(False, 2.0, 1.0, 3.0)
#     vertices2, _, triangles2, normals2 = build_tetrahedron()
#     df = pl.DataFrame(
#         {
#             "ID": ["octahedron", "tetrahedron"],
#             "simplices": {
#                 "vertices": [vertices.tolist(), vertices2.tolist()],
#                 "triangles": [triangles.tolist(), triangles2.tolist()],
#             },
#         },
#     )
#
#
#     ci = ComplexInfo(
#         simplices="simplices",
#         vdim=3,
#     )
#
#     ea = EctArgs(directions=25, steps=20)
#
#     print(df)
#
#     edf = with_ects(
#         df.lazy(),
#         ci,
#         ea=ea,
#         ename="ects",
#     ).select("ID", "simplices", "ects")
#     print(edf.explain(engine=True))
#     edf = edf.collect()
#
#     ects = edf.to_dict()["ects"].to_numpy()
#     for ect in ects:
#         print(ect.reshape(25, 20))
#
#
# def test_premapped_ect() -> None:
#     vertices, triangles, normals = build_octahedron(False, 2.0, 1.0, 3.0)
#     vertices2, _, triangles2, normals2 = build_tetrahedron()
#     df = pl.DataFrame(
#         {
#             "ID": ["octahedron", "tetrahedron"],
#             "simplices": {
#                 "vertices": [vertices.tolist(), vertices2.tolist()],
#                 "triangles": [triangles.tolist(), triangles2.tolist()],
#             },
#         },
#     )
#
#
#     ci = ComplexInfo(
#         simplices="simplices",
#         vdim=3,
#     )
#
#     ea = EctArgs(directions=25, steps=20)
#     ma = MapArgs(
#         subsample_ratio=1.0,
#         subsample_min=10,
#         subsample_max=100,
#         align_dimension=2,
#     )
#
#     print(df)
#
#     edf = with_premapped_ects(
#         df.lazy(),
#         ci,
#         ma=ma,
#         ea=ea,
#         ename="ects",
#     ).select("ID", "simplices", "ects")
#     print(edf.explain(engine=True))
#     edf = edf.collect()
#
#     ects = edf.to_dict()["ects"].to_numpy()
#     for ect in ects:
#         print(ect.reshape(25, 20))
#
#
# def test_premapped_copy_ect() -> None:
#     vertices, triangles, normals = build_octahedron(False, 2.0, 1.0, 3.0)
#     vertices2, _, triangles2, normals2 = build_tetrahedron()
#     df = pl.DataFrame(
#         {
#             "ID": ["octahedron", "tetrahedron"],
#             "simplices": {
#                 "vertices": [vertices.tolist(), vertices2.tolist()],
#                 "triangles": [triangles.tolist(), triangles2.tolist()],
#             },
#         },
#     )
#
#
#     ci = ComplexInfo(
#         simplices="simplices",
#         vdim=3,
#     )
#
#     ea = EctArgs(directions=25, steps=20)
#     ma = MapCopyArgs(
#         subsample_ratio=1.0,
#         subsample_min=10,
#         subsample_max=100,
#         eps=0.01,
#         copies=False,
#         align_dimension=2,
#     )
#
#     print(df)
#
#     edf = with_premapped_copy_ects(
#         df.lazy(),
#         ci,
#         ma=ma,
#         ea=ea,
#         ename="ects",
#     ).select("ID", "simplices", "ects")
#     print(edf.explain(engine=True))
#     edf = edf.collect()
#
#     ects = edf.to_dict()["ects"].to_numpy()
#     for ect in ects:
#         print(ect.reshape(25, 20))
