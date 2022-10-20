import sys
sys.path.append("../../../../")
import os
import pytest

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from fastNLP.core.drivers.oneflow_driver.dist_utils import (
    _tensor_to_object,
    _object_to_tensor,
    fastnlp_oneflow_all_gather,
    fastnlp_oneflow_broadcast_object,
)

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    import oneflow.comm as comm

# @pytest.mark.oneflow
# class TestDistUtilsTools:
#     """
#     测试一些工具函数
#     """
@pytest.mark.oneflow
@pytest.mark.parametrize("device", (["cpu", int(os.getenv("LOCAL_RANK", "0"))]))
def test_tensor_object_transfer_tensor(device):
    """
    测试 _tensor_to_object 和 _object_to_tensor 二者的结果能否互相转换
    """
    # 张量
    oneflow_tensor = oneflow.rand(3, 4, 5)
    obj_tensor, size = _object_to_tensor(oneflow_tensor, device=device)
    res = _tensor_to_object(obj_tensor, size)
    assert oneflow.all(res == oneflow_tensor)

    # 列表
    oneflow_list = [oneflow.rand(6, 4, 2) for i in range(10)]
    obj_tensor, size = _object_to_tensor(oneflow_list, device=device)
    res = _tensor_to_object(obj_tensor, size)
    assert isinstance(res, list)
    for before, after in zip(oneflow_list, res):
        assert oneflow.all(after == before)

    # 元组
    oneflow_list = [oneflow.rand(6, 4, 2) for i in range(10)]
    oneflow_tuple = tuple(oneflow_list)
    obj_tensor, size = _object_to_tensor(oneflow_tuple, device=device)
    res = _tensor_to_object(obj_tensor, size)
    assert isinstance(res, tuple)
    for before, after in zip(oneflow_list, res):
        assert oneflow.all(after == before)
        
    # 字典
    oneflow_dict = {
        "tensor": oneflow.rand(3, 4),
        "list": [oneflow.rand(6, 4, 2) for i in range(10)],
        "dict":{
            "list": [oneflow.rand(6, 4, 2) for i in range(10)],
            "tensor": oneflow.rand(3, 4)
        },
        "int": 2,
        "string": "test string"
    }
    obj_tensor, size = _object_to_tensor(oneflow_dict, device=device)
    res = _tensor_to_object(obj_tensor, size)
    assert isinstance(res, dict)
    assert oneflow.all(res["tensor"] == oneflow_dict["tensor"])
    assert isinstance(res["list"], list)
    for before, after in zip(oneflow_dict["list"], res["list"]):
        assert oneflow.all(after == before)

    assert isinstance(res["dict"], dict)
    assert oneflow.all(res["dict"]["tensor"] == oneflow_dict["dict"]["tensor"])
    for before, after in zip(oneflow_dict["dict"]["list"], res["dict"]["list"]):
        assert oneflow.all(after == before)
    assert res["int"] == oneflow_dict["int"]
    assert res["string"] == oneflow_dict["string"]

@pytest.mark.oneflowdist
def test_fastnlp_oneflow_all_gather():
    local_rank = int(os.environ["LOCAL_RANK"])
    obj = {
        "tensor": oneflow.full((2, ), local_rank, oneflow.int).cuda(),
        "numpy": np.full(shape=(2, ), fill_value=local_rank),
        "bool": local_rank % 2 == 0,
        "float": local_rank + 0.1,
        "int": local_rank,
        "dict": {
            "rank": local_rank
        },
        "list": [local_rank]*2,
        "str": f"{local_rank}",
        "tensors": [oneflow.full((2, ), local_rank, oneflow.int).cuda(),
                    oneflow.full((2, ), local_rank, oneflow.int).cuda()]
    }
    data = fastnlp_oneflow_all_gather(obj)
    world_size = int(os.environ["WORLD_SIZE"])
    assert len(data) == world_size
    for i in range(world_size):
        assert (data[i]["tensor"] == i).sum() == world_size
        assert data[i]["numpy"][0] == i
        assert data[i]["bool"] == (i % 2 == 0)
        assert np.allclose(data[i]["float"], i + 0.1)
        assert data[i]["int"] == i
        assert data[i]["dict"]["rank"] == i
        assert data[i]["list"][0] == i
        assert data[i]["str"] == f"{i}"
        assert data[i]["tensors"][0][0] == i

    for obj in [1, True, "xxx"]:
        data = fastnlp_oneflow_all_gather(obj)
        assert len(data) == world_size
        assert data[0] == data[1]

@pytest.mark.oneflowdist
def test_fastnlp_oneflow_broadcast_object():
    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ["LOCAL_RANK"] == "0":
        obj = {
            "tensor": oneflow.full((2, ), local_rank, oneflow.int).cuda(),
            "numpy": np.full(shape=(2, ), fill_value=local_rank, dtype=int),
            "bool": local_rank % 2 == 0,
            "float": local_rank + 0.1,
            "int": local_rank,
            "dict": {
                "rank": local_rank
            },
            "list": [local_rank] * 2,
            "str": f"{local_rank}",
            "tensors": [oneflow.full((2, ), local_rank, oneflow.int).cuda(),
                        oneflow.full((2, ), local_rank, oneflow.int).cuda()]
        }
    else:
        obj = None
    # device=oneflow.cuda.current_devuce
    data = fastnlp_oneflow_broadcast_object(obj, src=0, device=local_rank)
    i = 0
    assert data["tensor"][0] == 0
    assert data["numpy"][0] == 0
    assert data["bool"] == (i % 2 == 0)
    assert np.allclose(data["float"], i + 0.1)
    assert data["int"] == i
    assert data["dict"]["rank"] == i
    assert data["list"][0] == i
    assert data["str"] == f"{i}"
    assert data["tensors"][0][0] == i

    for obj in [local_rank, bool(local_rank== 1), str(local_rank)]:
        data = fastnlp_oneflow_broadcast_object(obj, src=0, device=local_rank)
        assert int(data) == 0

if __name__ == "__main__":
    # python -m oneflow.distributed.launch --nproc_per_node 2 test_dist_utils.py
    pytest.main([
        f'{__file__}'
    ])
