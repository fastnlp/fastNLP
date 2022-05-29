import pytest

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH
from fastNLP.modules.mix_modules.utils import (
    paddle2torch,
    torch2paddle,
    jittor2torch,
    torch2jittor,
)

if _NEED_IMPORT_TORCH:
    import torch

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_JITTOR:
    import jittor


############################################################################
#
# 测试paddle到torch的转换
#
############################################################################

@pytest.mark.torchpaddle
class TestPaddle2Torch:

    def check_torch_tensor(self, tensor, device, requires_grad):
        """
        检查张量设备和梯度情况的工具函数
        """

        assert isinstance(tensor, torch.Tensor)
        if device == "cpu":
            assert not tensor.is_cuda
        else:
            assert tensor.is_cuda
            assert tensor.device.index == torch.device(device).index
        assert tensor.requires_grad == requires_grad

    def test_gradient(self):
        """
        测试张量转换后的反向传播是否正确
        """

        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], stop_gradient=False)
        y = paddle2torch(x)
        z = 3 * (y ** 2)
        z.sum().backward()
        assert y.grad.tolist() == [6, 12, 18, 24, 30]

    def test_tensor_transfer(self):
        """
        测试单个张量的设备和梯度转换是否正确
        """

        paddle_tensor = paddle.rand((3, 4, 5)).cpu()
        res = paddle2torch(paddle_tensor)
        self.check_torch_tensor(res, "cpu", not paddle_tensor.stop_gradient)

        res = paddle2torch(paddle_tensor, device="cuda:2", no_gradient=None)
        self.check_torch_tensor(res, "cuda:2", not paddle_tensor.stop_gradient)

        res = paddle2torch(paddle_tensor, device="cuda:1", no_gradient=True)
        self.check_torch_tensor(res, "cuda:1", False)

        res = paddle2torch(paddle_tensor, device="cuda:1", no_gradient=False)
        self.check_torch_tensor(res, "cuda:1", True)

    def test_list_transfer(self):
        """
        测试张量列表的转换
        """

        paddle_list = [paddle.rand((6, 4, 2)).cuda(1) for i in range(10)]
        res = paddle2torch(paddle_list)
        assert isinstance(res, list)
        for t in res:
            self.check_torch_tensor(t, "cuda:1", False)

        res = paddle2torch(paddle_list, device="cpu", no_gradient=False)
        assert isinstance(res, list)
        for t in res:
            self.check_torch_tensor(t, "cpu", True)

    def test_tensor_tuple_transfer(self):
        """
        测试张量元组的转换
        """

        paddle_list = [paddle.rand((6, 4, 2)).cuda(1) for i in range(10)]
        paddle_tuple = tuple(paddle_list)
        res = paddle2torch(paddle_tuple)
        assert isinstance(res, tuple)
        for t in res:
            self.check_torch_tensor(t, "cuda:1", False)

    def test_dict_transfer(self):
        """
        测试包含复杂结构的字典的转换
        """

        paddle_dict = {
            "tensor": paddle.rand((3, 4)).cuda(0),
            "list": [paddle.rand((6, 4, 2)).cuda(0) for i in range(10)],
            "dict":{
                "list": [paddle.rand((6, 4, 2)).cuda(0) for i in range(10)],
                "tensor": paddle.rand((3, 4)).cuda(0)
            },
            "int": 2,
            "string": "test string"
        }
        res = paddle2torch(paddle_dict)
        assert isinstance(res, dict)
        self.check_torch_tensor(res["tensor"], "cuda:0", False)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_torch_tensor(t, "cuda:0", False)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_torch_tensor(t, "cuda:0", False)
        self.check_torch_tensor(res["dict"]["tensor"], "cuda:0", False)


############################################################################
#
# 测试torch到paddle的转换
#
############################################################################

@pytest.mark.torchpaddle
class TestTorch2Paddle:

    def check_paddle_tensor(self, tensor, device, stop_gradient):
        """
        检查得到的paddle张量设备和梯度情况的工具函数
        """

        assert isinstance(tensor, paddle.Tensor)
        if device == "cpu":
            assert tensor.place.is_cpu_place()
        elif device.startswith("gpu"):
            paddle_device = paddle.device._convert_to_place(device)
            assert tensor.place.is_gpu_place()
            if hasattr(tensor.place, "gpu_device_id"):
                # paddle中，有两种Place
                # paddle.fluid.core.Place是创建Tensor时使用的类型
                # 有函数gpu_device_id获取设备
                assert tensor.place.gpu_device_id() == paddle_device.get_device_id()
            else:
                # 通过_convert_to_place得到的是paddle.CUDAPlace
                # 通过get_device_id获取设备
                assert tensor.place.get_device_id() == paddle_device.get_device_id()
        else:
            raise NotImplementedError
        assert tensor.stop_gradient == stop_gradient

    def test_gradient(self):
        """
        测试转换后梯度的反向传播
        """

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = torch2paddle(x)
        z = 3 * (y ** 2)
        z.sum().backward()
        assert y.grad.tolist() == [6, 12, 18, 24, 30]

    def test_tensor_transfer(self):
        """
        测试单个张量的转换
        """

        torch_tensor = torch.rand((3, 4, 5))
        res = torch2paddle(torch_tensor)
        self.check_paddle_tensor(res, "cpu", True)

        res = torch2paddle(torch_tensor, device="gpu:2", no_gradient=None)
        self.check_paddle_tensor(res, "gpu:2", True)

        res = torch2paddle(torch_tensor, device="gpu:2", no_gradient=True)
        self.check_paddle_tensor(res, "gpu:2", True)

        res = torch2paddle(torch_tensor, device="gpu:2", no_gradient=False)
        self.check_paddle_tensor(res, "gpu:2", False)

    def test_tensor_list_transfer(self):
        """
        测试张量列表的转换
        """

        torch_list = [torch.rand(6, 4, 2) for i in range(10)]
        res = torch2paddle(torch_list)
        assert isinstance(res, list)
        for t in res:
            self.check_paddle_tensor(t, "cpu", True)

        res = torch2paddle(torch_list, device="gpu:1", no_gradient=False)
        assert isinstance(res, list)
        for t in res:
            self.check_paddle_tensor(t, "gpu:1", False)

    def test_tensor_tuple_transfer(self):
        """
        测试张量元组的转换
        """
        
        torch_list = [torch.rand(6, 4, 2) for i in range(10)]
        torch_tuple = tuple(torch_list)
        res = torch2paddle(torch_tuple, device="cpu")
        assert isinstance(res, tuple)
        for t in res:
            self.check_paddle_tensor(t, "cpu", True)

    def test_dict_transfer(self):
        """
        测试复杂的字典结构的转换
        """

        torch_dict = {
            "tensor": torch.rand((3, 4)),
            "list": [torch.rand(6, 4, 2) for i in range(10)],
            "dict":{
                "list": [torch.rand(6, 4, 2) for i in range(10)],
                "tensor": torch.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }
        res = torch2paddle(torch_dict)
        assert isinstance(res, dict)
        self.check_paddle_tensor(res["tensor"], "cpu", True)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_paddle_tensor(t, "cpu", True)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_paddle_tensor(t, "cpu", True)
        self.check_paddle_tensor(res["dict"]["tensor"], "cpu", True)


############################################################################
#
# 测试jittor到torch的转换
#
############################################################################

@pytest.mark.torchjittor
class TestJittor2Torch:

    def check_torch_tensor(self, tensor, device, requires_grad):
        """
        检查得到的torch张量的工具函数
        """

        assert isinstance(tensor, torch.Tensor)
        if device == "cpu":
            assert not tensor.is_cuda
        else:
            assert tensor.is_cuda
            assert tensor.device.index == torch.device(device).index
        assert tensor.requires_grad == requires_grad

    def test_var_transfer(self):
        """
        测试单个Jittor Var的转换
        """

        jittor_var = jittor.rand((3, 4, 5))
        res = jittor2torch(jittor_var)
        if jittor.flags.use_cuda:
            self.check_torch_tensor(res, "cuda:0", True)
        else:
            self.check_torch_tensor(res, "cpu", True)

        res = jittor2torch(jittor_var, device="cuda:2", no_gradient=None)
        self.check_torch_tensor(res, "cuda:2", True)

        res = jittor2torch(jittor_var, device="cuda:2", no_gradient=True)
        self.check_torch_tensor(res, "cuda:2", False)

        res = jittor2torch(jittor_var, device="cuda:2", no_gradient=False)
        self.check_torch_tensor(res, "cuda:2", True)

    def test_var_list_transfer(self):
        """
        测试Jittor列表的转换
        """

        jittor_list = [jittor.rand((6, 4, 2)) for i in range(10)]
        res = jittor2torch(jittor_list)
        assert isinstance(res, list)
        for t in res:
            if jittor.flags.use_cuda:
                self.check_torch_tensor(t, "cuda:0", True)
            else:
                self.check_torch_tensor(t, "cpu", True)

        res = jittor2torch(jittor_list, device="cuda:1", no_gradient=False)
        assert isinstance(res, list)
        for t in res:
            self.check_torch_tensor(t, "cuda:1", True)

    def test_var_tuple_transfer(self):
        """
        测试Jittor变量元组的转换
        """

        jittor_list = [jittor.rand((6, 4, 2)) for i in range(10)]
        jittor_tuple = tuple(jittor_list)
        res = jittor2torch(jittor_tuple, device="cpu")
        assert isinstance(res, tuple)
        for t in res:
            self.check_torch_tensor(t, "cpu", True)

    def test_dict_transfer(self):
        """
        测试字典结构的转换
        """

        jittor_dict = {
            "tensor": jittor.rand((3, 4)),
            "list": [jittor.rand(6, 4, 2) for i in range(10)],
            "dict":{
                "list": [jittor.rand(6, 4, 2) for i in range(10)],
                "tensor": jittor.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }
        res = jittor2torch(jittor_dict)
        assert isinstance(res, dict)
        if jittor.flags.use_cuda:
            self.check_torch_tensor(res["tensor"], "cuda:0", True)
        else:
            self.check_torch_tensor(res["tensor"], "cpu", True)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            if jittor.flags.use_cuda:
                self.check_torch_tensor(t, "cuda:0", True)
            else:
                self.check_torch_tensor(t, "cpu", True)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            if jittor.flags.use_cuda:
                self.check_torch_tensor(t, "cuda:0", True)
            else:
                self.check_torch_tensor(t, "cpu", True)
        if jittor.flags.use_cuda:
            self.check_torch_tensor(res["dict"]["tensor"], "cuda:0", True)
        else:
            self.check_torch_tensor(res["dict"]["tensor"], "cpu", True)


############################################################################
#
# 测试torch到jittor的转换
#
############################################################################

@pytest.mark.torchjittor
class TestTorch2Jittor:

    def check_jittor_var(self, var, requires_grad):
        """
        检查得到的Jittor Var梯度情况的工具函数
        """

        assert isinstance(var, jittor.Var)
        assert var.requires_grad == requires_grad

    def test_gradient(self):
        """
        测试反向传播的梯度
        """

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = torch2jittor(x)
        z = 3 * (y ** 2)
        grad = jittor.grad(z, y)
        assert grad.tolist() == [6.0, 12.0, 18.0, 24.0, 30.0]

    def test_tensor_transfer(self):
        """
        测试单个张量转换为Jittor
        """

        torch_tensor = torch.rand((3, 4, 5))
        res = torch2jittor(torch_tensor)
        self.check_jittor_var(res, False)

        res = torch2jittor(torch_tensor, no_gradient=None)
        self.check_jittor_var(res, False)

        res = torch2jittor(torch_tensor, no_gradient=True)
        self.check_jittor_var(res, False)

        res = torch2jittor(torch_tensor, no_gradient=False)
        self.check_jittor_var(res, True)

    def test_tensor_list_transfer(self):
        """
        测试张量列表的转换
        """

        torch_list = [torch.rand((6, 4, 2)) for i in range(10)]
        res = torch2jittor(torch_list)
        assert isinstance(res, list)
        for t in res:
            self.check_jittor_var(t, False)

        res = torch2jittor(torch_list, no_gradient=False)
        assert isinstance(res, list)
        for t in res:
            self.check_jittor_var(t, True)

    def test_tensor_tuple_transfer(self):
        """
        测试张量元组的转换
        """

        torch_list = [torch.rand((6, 4, 2)) for i in range(10)]
        torch_tuple = tuple(torch_list)
        res = torch2jittor(torch_tuple)
        assert isinstance(res, tuple)
        for t in res:
            self.check_jittor_var(t, False)

    def test_dict_transfer(self):
        """
        测试字典结构的转换
        """

        torch_dict = {
            "tensor": torch.rand((3, 4)),
            "list": [torch.rand(6, 4, 2) for i in range(10)],
            "dict":{
                "list": [torch.rand(6, 4, 2) for i in range(10)],
                "tensor": torch.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }
        res = torch2jittor(torch_dict)
        assert isinstance(res, dict)
        self.check_jittor_var(res["tensor"], False)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_jittor_var(t, False)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_jittor_var(t, False)
        self.check_jittor_var(res["dict"]["tensor"], False)
