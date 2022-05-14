import os

import pytest

from fastNLP.core.utils.paddle_utils import _convert_data_device, paddle_to, paddle_move_data_to_device
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
if _NEED_IMPORT_PADDLE:
    import paddle

@pytest.mark.parametrize(
    ("user_visible_devices, cuda_visible_devices, device, correct"),
    (
        (None, None, 1, "gpu:1"),
        (None, "2,4,5,6", 2, "gpu:2"),
        (None, "3,4,5", 1, "gpu:1"),
        ("0,1,2,3,4,5,6,7", "0", "cpu", "cpu"),
        ("3,4,5,6,7", "0", "cpu", "cpu"),
        ("0,1,2,3,4,5,6,7", "3,4,5", "gpu:4", "gpu:1"),
        ("0,1,2,3,4,5,6,7", "3,4,5", "gpu:5", "gpu:2"),
        ("3,4,5,6", "3,5", 0, "gpu:0"),
        ("3,6,7,8", "6,7,8", "gpu:2", "gpu:1"),
    )
)
def test_convert_data_device(user_visible_devices, cuda_visible_devices, device, correct):
    _cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    _user_visible_devices = os.getenv("USER_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if user_visible_devices is not None:
        os.environ["USER_CUDA_VISIBLE_DEVICES"] = user_visible_devices
    res = _convert_data_device(device)
    assert res == correct

    # 还原环境变量
    if _cuda_visible_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices
    if _user_visible_devices is None:
        os.environ.pop("USER_CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["USER_CUDA_VISIBLE_DEVICES"] = _user_visible_devices

############################################################################
#
# 测试仅将单个paddle张量迁移到指定设备
#
############################################################################

@pytest.mark.paddle
class TestPaddleToDevice:
    def test_case(self):
        tensor = paddle.rand((4, 5))

        res = paddle_to(tensor, "gpu")
        assert res.place.is_gpu_place()
        assert res.place.gpu_device_id() == 0
        res = paddle_to(tensor, "cpu")
        assert res.place.is_cpu_place()

############################################################################
#
# 测试将参数中包含的所有paddle张量迁移到指定设备
#
############################################################################

class TestPaddleMoveDataToDevice:

    def check_gpu(self, tensor, idx):
        """
        检查张量是否在指定的设备上的工具函数
        """

        assert tensor.place.is_gpu_place()
        assert tensor.place.gpu_device_id() == idx

    def check_cpu(self, tensor):
        """
        检查张量是否在cpu上的工具函数
        """

        assert tensor.place.is_cpu_place()

    def test_tensor_transfer(self):
        """
        测试单个张量的迁移
        """

        paddle_tensor = paddle.rand((3, 4, 5)).cpu()
        res = paddle_move_data_to_device(paddle_tensor, device=None)
        self.check_cpu(res)

        res = paddle_move_data_to_device(paddle_tensor, device="gpu:0")
        self.check_gpu(res, 0)

    def test_list_transfer(self):
        """
        测试张量列表的迁移
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]

        res = paddle_move_data_to_device(paddle_list, device="cpu")
        assert isinstance(res, list)
        for r in res:
            self.check_cpu(r)

        res = paddle_move_data_to_device(paddle_list, device="gpu:0")
        assert isinstance(res, list)
        for r in res:
            self.check_gpu(r, 0)

    def test_tensor_tuple_transfer(self):
        """
        测试张量元组的迁移
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        paddle_tuple = tuple(paddle_list)

        res = paddle_move_data_to_device(paddle_tuple, device="cpu")
        assert isinstance(res, tuple)
        for r in res:
            self.check_cpu(r)

        res = paddle_move_data_to_device(paddle_tuple, device="gpu:0")
        assert isinstance(res, tuple)
        for r in res:
            self.check_gpu(r, 0)

    def test_dict_transfer(self):
        """
        测试字典结构的迁移
        """

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

        res = paddle_move_data_to_device(paddle_dict, device="gpu:0")
        assert isinstance(res, dict)
        self.check_gpu(res["tensor"], 0)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 0)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 0)
        self.check_gpu(res["dict"]["tensor"], 0)

        res = paddle_move_data_to_device(paddle_dict, device="gpu:0")
        assert isinstance(res, dict)
        self.check_gpu(res["tensor"], 0)
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 0)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 0)
        self.check_gpu(res["dict"]["tensor"], 0)

        res = paddle_move_data_to_device(paddle_dict, device="cpu")
        assert isinstance(res, dict)
        self.check_cpu(res["tensor"])
        assert isinstance(res["list"], list)
        for t in res["list"]:
            self.check_cpu(t)
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_cpu(t)
        self.check_cpu(res["dict"]["tensor"])
