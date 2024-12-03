import polars as pl
from tdataframe import loading as tdld
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, "test_data")


def test_load_objs() -> None:
    loader = tdld.WeightedObj()
    obj_dir = os.path.join(TEST_DATA_DIR, "objs")
    # get list of files in the directory
    files = [
        os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith(".obj")
    ]
    df = loader.load_complexes(files)
    print(df)


def test_load_graphs() -> None:
    loader = tdld.GraphMl()
    obj_dir = os.path.join(TEST_DATA_DIR, "graphml")
    # get list of files in the directory
    files = [
        os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith(".graphml")
    ]
    df = loader.load_complexes(files)
    print(df)


if __name__ == "__main__":
    print(THIS_DIR)
    # test_load_objs()
    test_load_graphs()
