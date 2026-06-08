import numpy as np
import polars as pl

from tdataframe.ect import with_wects
from tdataframe.params import EctArgs, WeightedComplexInfo
from tdataframe.retrieval import exact_knn, l1_distance, l2_distance


def build_octahedron(weighted: bool, x: float = 1.0, y: float = 1.0, z: float = 1.0):
    vertices = np.array(
        [
            [x, 0, 0],
            [-x, 0, 0],
            [0, y, 0],
            [0, -y, 0],
            [0, 0, z],
            [0, 0, -z],
        ],
        dtype=np.float32,
    )
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
    return vertices, faces, non_uniform if weighted else uniform


def test_distance_helpers() -> None:
    left = [0.0, 3.0, 4.0]
    right = [0.0, 0.0, 0.0]

    assert l2_distance(left, right) == 5.0
    assert l1_distance(left, right) == 7.0


def test_exact_knn_on_plain_vectors() -> None:
    df = pl.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "descriptor": [[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]],
        }
    )

    out = exact_knn(df, [0.0, 0.0], "descriptor", k=2)

    assert out.get_column("ID").to_list() == ["a", "b"]
    assert out.get_column("distance").to_list()[0] == 0.0


def test_exact_knn_on_wects() -> None:
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    vertices_scaled, triangles_scaled, normals_scaled = build_octahedron(
        True, 3.0, 1.5, 4.0
    )
    df = pl.DataFrame(
        {
            "ID": ["octahedron", "scaled"],
            "simplices": {
                "vertices": [vertices.tolist(), vertices_scaled.tolist()],
                "triangles": [triangles.tolist(), triangles_scaled.tolist()],
            },
            "weights": {
                "trinormals": [normals.tolist(), normals_scaled.tolist()],
            },
        }
    )

    wci = WeightedComplexInfo(
        simplices="simplices",
        weights="weights",
        provided_weights=[2],
    )
    ea = EctArgs(directions=16, steps=12)

    wdf = (
        with_wects(df.lazy(), wci, ea=ea, wname="wects")
        .select("ID", "wects")
        .collect()
    )

    query = wdf.get_column("wects").to_list()[0]
    out = exact_knn(wdf, query, "wects", k=2)

    assert out.get_column("ID").to_list()[0] == "octahedron"
    assert out.get_column("distance").to_list()[0] == 0.0
