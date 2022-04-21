import os
import sys
import signal
import pytest
import traceback
os.environ["FASTNLP_BACKEND"] = "paddle"

import numpy as np

from fastNLP.core.drivers.paddle_driver.dist_utils import (
    _tensor_to_object,
    _object_to_tensor,
    fastnlp_paddle_all_gather,
    fastnlp_paddle_broadcast_object,
)
from fastNLP.core.drivers.paddle_driver.fleet_launcher import FleetLauncher
from tests.helpers.utils import magic_argv_env_context

import paddle
import paddle.distributed as dist

class TestDistUtilsTools:
    """
    测试一些工具函数
    """

    @pytest.mark.parametrize("device", (["cpu", 0]))
    def test_tensor_object_transfer_tensor(self, device):
        """
        测试 _tensor_to_object 和 _object_to_tensor 二者的结果能否互相转换
        """
        # 张量
        paddle_tensor = paddle.rand((3, 4, 5)).cpu()
        obj_tensor, size = _object_to_tensor(paddle_tensor, device=device)
        res = _tensor_to_object(obj_tensor, size)
        assert paddle.equal_all(res, paddle_tensor)

        # 列表
        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        obj_tensor, size = _object_to_tensor(paddle_list, device=device)
        res = _tensor_to_object(obj_tensor, size)
        assert isinstance(res, list)
        for before, after in zip(paddle_list, res):
            assert paddle.equal_all(after, before)

        # 元组
        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        paddle_tuple = tuple(paddle_list)
        obj_tensor, size = _object_to_tensor(paddle_tuple, device=device)
        res = _tensor_to_object(obj_tensor, size)
        assert isinstance(res, tuple)
        for before, after in zip(paddle_list, res):
            assert paddle.equal_all(after, before)
            
        # 字典
        paddle_dict = {
            "tensor": paddle.rand((3, 4)),
            "list": [paddle.rand((6, 4, 2)) for i in range(10)],
            "dict":{
                "list": [paddle.rand((6, 4, 2)) for i in range(10)],
                "tensor": paddle.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }
        obj_tensor, size = _object_to_tensor(paddle_dict, device=device)
        res = _tensor_to_object(obj_tensor, size)
        assert isinstance(res, dict)
        assert paddle.equal_all(res["tensor"], paddle_dict["tensor"])
        assert isinstance(res["list"], list)
        for before, after in zip(paddle_dict["list"], res["list"]):
            assert paddle.equal_all(after, before)

        assert isinstance(res["dict"], dict)
        assert paddle.equal_all(res["dict"]["tensor"], paddle_dict["dict"]["tensor"])
        for before, after in zip(paddle_dict["dict"]["list"], res["dict"]["list"]):
            assert paddle.equal_all(after, before)
        assert res["int"] == paddle_dict["int"]
        assert res["string"] == paddle_dict["string"]


class TestAllGatherAndBroadCast:

    @classmethod
    def setup_class(cls):
        devices = [0,1,2]
        output_from_new_proc = "only_error"

        launcher = FleetLauncher(devices=devices, output_from_new_proc=output_from_new_proc)
        cls.local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", "0"))
        if cls.local_rank == 0:
            launcher = FleetLauncher(devices, output_from_new_proc)
            launcher.launch()
        dist.fleet.init(is_collective=True)
        dist.barrier()

        # cls._pids = []
        # dist.all_gather(cls._pids, paddle.to_tensor(os.getpid(), dtype="int32"))
        # local_world_size = paddle.to_tensor(cls.local_rank, dtype="int32")
        # dist.all_reduce(local_world_size, op=dist.ReduceOp.MAX)
        # local_world_size = local_world_size.item() + 1

    def on_exception(self):
        if self._pids is not None:

            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj, file=sys.stderr)
            sys.stderr.write(f"Start to stop these pids:{self._pids}, please wait several seconds.\n")
            for pid in self._pids:
                pid = pid.item()
                if pid != os.getpid():
                    os.kill(pid, signal.SIGKILL)

    @magic_argv_env_context
    def test_fastnlp_paddle_all_gather(self):
        obj = {
            'tensor': paddle.full(shape=(2, ), fill_value=self.local_rank).cuda(),
            'numpy': np.full(shape=(2, ), fill_value=self.local_rank),
            'bool': self.local_rank % 2 == 0,
            'float': self.local_rank + 0.1,
            'int': self.local_rank,
            'dict': {
                'rank': self.local_rank
            },
            'list': [self.local_rank] * 2,
            'str': f'{self.local_rank}',
            'tensors': [paddle.full(shape=(2,), fill_value=self.local_rank).cuda(),
                        paddle.full(shape=(2,), fill_value=self.local_rank).cuda()]
        }
        data = fastnlp_paddle_all_gather(obj)
        world_size = int(os.environ['PADDLE_TRAINERS_NUM'])
        assert len(data) == world_size
        for i in range(world_size):
            assert (data[i]['tensor'] == i).sum() == 2
            assert (data[i]['numpy'] == i).sum() == 2
            assert data[i]['bool'] == (i % 2 == 0)
            assert np.allclose(data[i]['float'], i + 0.1)
            assert data[i]['int'] == i
            assert data[i]['dict']['rank'] == i
            assert data[i]['list'][0] == i
            assert data[i]['str'] == f'{i}'
            assert data[i]['tensors'][0][0] == i

        for obj in [1, True, 'xxx']:
            data = fastnlp_paddle_all_gather(obj)
            assert len(data) == world_size
            assert data[0] == data[1]

        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("src_rank", ([0, 1, 2]))
    def test_fastnlp_paddle_broadcast_object(self, src_rank):
        if self.local_rank == src_rank:
            obj = {
                'tensor': paddle.full(shape=(2, ), fill_value=self.local_rank).cuda(),
                'numpy': np.full(shape=(2, ), fill_value=self.local_rank),
                'bool': self.local_rank % 2 == 0,
                'float': self.local_rank + 0.1,
                'int': self.local_rank,
                'dict': {
                    'rank': self.local_rank
                },
                'list': [self.local_rank] * 2,
                'str': f'{self.local_rank}',
                'tensors': [paddle.full(shape=(2,), fill_value=self.local_rank).cuda(),
                            paddle.full(shape=(2,), fill_value=self.local_rank).cuda()]
            }
        else:
            obj = None
        data = fastnlp_paddle_broadcast_object(obj, src=src_rank, device=paddle.device.get_device())
        assert data['tensor'][0] == src_rank
        assert data['numpy'][0] == src_rank
        assert data['bool'] == (src_rank % 2 == 0)
        assert np.allclose(data['float'], src_rank + 0.1)
        assert data['int'] == src_rank
        assert data['dict']['rank'] == src_rank
        assert data['list'][0] == src_rank
        assert data['str'] == f'{src_rank}'
        assert data['tensors'][0][0] == src_rank

        for obj in [self.local_rank, bool(self.local_rank == 1), str(self.local_rank)]:
            data = fastnlp_paddle_broadcast_object(obj, src=0, device=paddle.device.get_device())
            assert int (data) == 0
        dist.barrier()