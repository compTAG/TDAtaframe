import numpy as np
import polars as pl
import pytest

from tdataframe.building import build_descriptor_database
from tdataframe.params import EctArgs, WeightedComplexInfo


def build_octahedron(
    weighted: bool,
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
):
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


def test_build_descriptor_database_for_ect() -> None:
    vertices, triangles, _ = build_octahedron(False, 2.0, 1.0, 3.0)
    df = pl.DataFrame(
        {
            "ID": ["octahedron"],
            "family": ["octahedron"],
            "simplices": {
                "vertices": [vertices.tolist()],
                "triangles": [triangles.tolist()],
            },
        }
    )

    out = build_descriptor_database(
        df,
        descriptor="ect",
        ect_args=EctArgs(directions=16, steps=12),
        keep_columns=["family"],
    ).collect()

    assert out.columns == ["ID", "family", "descriptor"]
    assert out.height == 1
    assert len(out.get_column("descriptor")[0]) == 16 * 12


def test_build_descriptor_database_for_wect() -> None:
    vertices, triangles, normals = build_octahedron(True, 2.0, 1.0, 3.0)
    df = pl.DataFrame(
        {
            "ID": ["octahedron"],
            "family": ["octahedron"],
            "simplices": {
                "vertices": [vertices.tolist()],
                "triangles": [triangles.tolist()],
            },
            "weights": {
                "trinormals": [normals.tolist()],
            },
        }
    )

    out = build_descriptor_database(
        df,
        descriptor="wect",
        ect_args=EctArgs(directions=16, steps=12),
        wci=WeightedComplexInfo(
            simplices="simplices",
            weights="weights",
            provided_weights=[2],
        ),
        keep_columns=["family"],
    ).collect()

    assert out.columns == ["ID", "family", "descriptor"]
    assert out.height == 1
    assert len(out.get_column("descriptor")[0]) == 16 * 12


def test_build_descriptor_database_requires_wci_for_wect() -> None:
    df = pl.DataFrame({"ID": ["a"], "simplices": [{"vertices": [], "triangles": []}]})

    with pytest.raises(ValueError, match="wci is required"):
        build_descriptor_database(
            df,
            descriptor="wect",
            ect_args=EctArgs(directions=16, steps=12),
        )
