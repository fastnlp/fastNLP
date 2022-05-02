import paddle
import pytest
import torch

from fastNLP.core.utils.torch_paddle_utils import torch_paddle_move_data_to_device

############################################################################
#
# 测试将参数中包含的所有torch和paddle张量迁移到指定设备
#
############################################################################

@pytest.mark.torchpaddle
class TestTorchPaddleMoveDataToDevice:

    def check_gpu(self, tensor, idx):
        """
        检查张量是否在指定显卡上的工具函数
        """

        if isinstance(tensor, paddle.Tensor):
            assert tensor.place.is_gpu_place()
            assert tensor.place.gpu_device_id() == idx
        elif isinstance(tensor, torch.Tensor):
            assert tensor.is_cuda
            assert tensor.device.index == idx

    def check_cpu(self, tensor):
        if isinstance(tensor, paddle.Tensor):
            assert tensor.place.is_cpu_place()
        elif isinstance(tensor, torch.Tensor):
            assert not tensor.is_cuda

    def test_tensor_transfer(self):
        """
        测试迁移单个张量
        """

        paddle_tensor = paddle.rand((3, 4, 5)).cpu()
        res = torch_paddle_move_data_to_device(paddle_tensor, device=None, data_device=None)
        self.check_cpu(res)

        res = torch_paddle_move_data_to_device(paddle_tensor, device="gpu:0", data_device=None)
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(paddle_tensor, device="gpu:1", data_device=None)
        self.check_gpu(res, 1)

        res = torch_paddle_move_data_to_device(paddle_tensor, device="cuda:0", data_device="cpu")
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(paddle_tensor, device=None, data_device="gpu:0")
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(paddle_tensor, device=None, data_device="cuda:1")
        self.check_gpu(res, 1)

        torch_tensor = torch.rand(3, 4, 5)
        res = torch_paddle_move_data_to_device(torch_tensor, device=None, data_device=None)
        self.check_cpu(res)

        res = torch_paddle_move_data_to_device(torch_tensor, device="gpu:0", data_device=None)
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(torch_tensor, device="gpu:1", data_device=None)
        self.check_gpu(res, 1)

        res = torch_paddle_move_data_to_device(torch_tensor, device="gpu:0", data_device="cpu")
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(torch_tensor, device=None, data_device="gpu:0")
        self.check_gpu(res, 0)

        res = torch_paddle_move_data_to_device(torch_tensor, device=None, data_device="gpu:1")
        self.check_gpu(res, 1)

    def test_list_transfer(self):
        """
        测试迁移张量的列表
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(5)] + [torch.rand((6, 4, 2)) for i in range(5)]
        res = torch_paddle_move_data_to_device(paddle_list, device=None, data_device="gpu:1")
        assert isinstance(res, list)
        for r in res:
            self.check_gpu(r, 1)

        res = torch_paddle_move_data_to_device(paddle_list, device="cpu", data_device="gpu:1")
        assert isinstance(res, list)
        for r in res:
            self.check_cpu(r)

        res = torch_paddle_move_data_to_device(paddle_list, device="gpu:0", data_device=None)
        assert isinstance(res, list)
        for r in res:
            self.check_gpu(r, 0)

        res = torch_paddle_move_data_to_device(paddle_list, device="gpu:1", data_device="cpu")
        assert isinstance(res, list)
        for r in res:
            self.check_gpu(r, 1)

    def test_tensor_tuple_transfer(self):
        """
        测试迁移张量的元组
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)] + [torch.rand((6, 4, 2)) for i in range(5)]
        paddle_tuple = tuple(paddle_list)
        res = torch_paddle_move_data_to_device(paddle_tuple, device=None, data_device="gpu:1")
        assert isinstance(res, tuple)
        for r in res:
            self.check_gpu(r, 1)

        res = torch_paddle_move_data_to_device(paddle_tuple, device="cpu", data_device="gpu:1")
        assert isinstance(res, tuple)
        for r in res:
            self.check_cpu(r)

        res = torch_paddle_move_data_to_device(paddle_tuple, device="gpu:0", data_device=None)
        assert isinstance(res, tuple)
        for r in res:
            self.check_gpu(r, 0)

        res = torch_paddle_move_data_to_device(paddle_tuple, device="gpu:1", data_device="cpu")
        assert isinstance(res, tuple)
        for r in res:
            self.check_gpu(r, 1)

    def test_dict_transfer(self):
        """
        测试迁移复杂的字典结构
        """

        paddle_dict = {
            "torch_tensor": torch.rand((3, 4)),
            "torch_list": [torch.rand((6, 4, 2)) for i in range(10)],
            "dict":{
                "list": [paddle.rand((6, 4, 2)) for i in range(5)] + [torch.rand((6, 4, 2)) for i in range(5)],
                "torch_tensor": torch.rand((3, 4)),
                "paddle_tensor": paddle.rand((3, 4))
            },
            "paddle_tensor": paddle.rand((3, 4)),
            "list": [paddle.rand((6, 4, 2)) for i in range(10)] ,
            "int": 2,
            "string": "test string"
        }

        res = torch_paddle_move_data_to_device(paddle_dict, device="gpu:0", data_device=None)
        assert isinstance(res, dict)
        self.check_gpu(res["torch_tensor"], 0)
        self.check_gpu(res["paddle_tensor"], 0)
        assert isinstance(res["torch_list"], list)
        for t in res["torch_list"]:
            self.check_gpu(t, 0)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 0)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 0)
        self.check_gpu(res["dict"]["torch_tensor"], 0)
        self.check_gpu(res["dict"]["paddle_tensor"], 0)

        res = torch_paddle_move_data_to_device(paddle_dict, device=None, data_device="gpu:1")
        assert isinstance(res, dict)
        self.check_gpu(res["torch_tensor"], 1)
        self.check_gpu(res["paddle_tensor"], 1)
        assert isinstance(res["torch_list"], list)
        for t in res["torch_list"]:
            self.check_gpu(t, 1)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 1)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 1)
        self.check_gpu(res["dict"]["torch_tensor"], 1)
        self.check_gpu(res["dict"]["paddle_tensor"], 1)

        res = torch_paddle_move_data_to_device(paddle_dict, device="cpu", data_device="gpu:0")
        assert isinstance(res, dict)
        self.check_cpu(res["torch_tensor"])
        self.check_cpu(res["paddle_tensor"])
        assert isinstance(res["torch_list"], list)
        for t in res["torch_list"]:
            self.check_cpu(t)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_cpu(t)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_cpu(t)
        self.check_cpu(res["dict"]["torch_tensor"])
        self.check_cpu(res["dict"]["paddle_tensor"])
