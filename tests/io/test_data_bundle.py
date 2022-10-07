import pytest

from fastNLP.core.dataset import DataSet
from fastNLP.io.data_bundle import DataBundle

def test_add_seq_len():
    dataset1 = DataSet({
        "x": [[0,1,2], [5,3,2,3], [5,21,5,10], [3,6,8,1]]
    })
    dataset2 = DataSet({
        "x": [[0,1,2,3,4], [5,3,2,3], [5,20,45,1,98], [3,6,8,3,6,31]]
    })
    dataset3 = DataSet({
        "x": [[0,1,2,7,5,2], [5,3], [0], [3,6,8]]
    })
    data_bundle = DataBundle(datasets={
        "dataset1": dataset1,
        "dataset2": dataset2,
        "dataset3": dataset3
    })
    data_bundle.add_seq_len("x")
    print(data_bundle.get_dataset("dataset1"))
    for i, data in enumerate(data_bundle.get_dataset("dataset1")):
        print(data["seq_len"], dataset1["x"][i])
        assert data["seq_len"] == len(dataset1["x"][i])
    for i, data in enumerate(data_bundle.get_dataset("dataset2")):
        assert data["seq_len"] == len(dataset2["x"][i])
    for i, data in enumerate(data_bundle.get_dataset("dataset3")):
        assert data["seq_len"] == len(dataset3["x"][i])

@pytest.mark.parametrize("inplace", [True, False])
def test_drop(inplace):
    dataset1 = DataSet({
        "x": [0, 1, 1, 4, 2, 1, 0, 1, 1, 6, 7, 1]
    })
    dataset2 = DataSet({
        "x": [0, 0, 0, 0, 0]
    })
    dataset3 = DataSet({
        "x": [1, 1, 1, 1, 1, 2, 3, 4]
    })
    data_bundle = DataBundle(datasets={
        "dataset1": dataset1,
        "dataset2": dataset2,
        "dataset3": dataset3
    })
    res = data_bundle.drop(lambda x: x["x"] == 0, inplace)
    if inplace:
        assert res is data_bundle
    else:
        assert not (res is data_bundle)
        assert data_bundle.get_dataset("dataset1")["x"] == dataset1["x"]
        assert data_bundle.get_dataset("dataset2")["x"] == dataset2["x"]
        assert data_bundle.get_dataset("dataset3")["x"] == dataset3["x"]

    dataset1_drop = [1, 1, 4, 2, 1, 1, 1, 6, 7, 1]
    for i, data in enumerate(res.get_dataset("dataset1")["x"]):
        assert data == dataset1_drop[i]
    dataset2_drop = []
    for i, data in enumerate(res.get_dataset("dataset2")["x"]):
        assert data == dataset2_drop[i]
    dataset3_drop = [1, 1, 1, 1, 1, 2, 3, 4]
    for i, data in enumerate(res.get_dataset("dataset3")["x"]):
        assert data == dataset3_drop[i]