import unittest

import paddle

from fastNLP.core.utils.paddle_utils import paddle_to, paddle_move_data_to_device


############################################################################
#
# 测试仅将单个paddle张量迁移到指定设备
#
############################################################################

class PaddleToDeviceTestCase(unittest.TestCase):
    def test_case(self):
        tensor = paddle.rand((4, 5))

        res = paddle_to(tensor, "gpu")
        self.assertTrue(res.place.is_gpu_place())
        self.assertEqual(res.place.gpu_device_id(), 0)
        res = paddle_to(tensor, "cpu")
        self.assertTrue(res.place.is_cpu_place())
        res = paddle_to(tensor, "gpu:2")
        self.assertTrue(res.place.is_gpu_place())
        self.assertEqual(res.place.gpu_device_id(), 2)
        res = paddle_to(tensor, "gpu:1")
        self.assertTrue(res.place.is_gpu_place())
        self.assertEqual(res.place.gpu_device_id(), 1)

############################################################################
#
# 测试将参数中包含的所有paddle张量迁移到指定设备
#
############################################################################

class PaddleMoveDataToDeviceTestCase(unittest.TestCase):

    def check_gpu(self, tensor, idx):
        """
        检查张量是否在指定的设备上的工具函数
        """

        self.assertTrue(tensor.place.is_gpu_place())
        self.assertEqual(tensor.place.gpu_device_id(), idx)

    def check_cpu(self, tensor):
        """
        检查张量是否在cpu上的工具函数
        """

        self.assertTrue(tensor.place.is_cpu_place())

    def test_tensor_transfer(self):
        """
        测试单个张量的迁移
        """

        paddle_tensor = paddle.rand((3, 4, 5)).cpu()
        res = paddle_move_data_to_device(paddle_tensor, device=None, data_device=None)
        self.check_cpu(res)

        res = paddle_move_data_to_device(paddle_tensor, device="gpu:0", data_device=None)
        self.check_gpu(res, 0)

        res = paddle_move_data_to_device(paddle_tensor, device="gpu:1", data_device=None)
        self.check_gpu(res, 1)

        res = paddle_move_data_to_device(paddle_tensor, device="gpu:0", data_device="cpu")
        self.check_gpu(res, 0)

        res = paddle_move_data_to_device(paddle_tensor, device=None, data_device="gpu:0")
        self.check_gpu(res, 0)

        res = paddle_move_data_to_device(paddle_tensor, device=None, data_device="gpu:1")
        self.check_gpu(res, 1)

    def test_list_transfer(self):
        """
        测试张量列表的迁移
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        res = paddle_move_data_to_device(paddle_list, device=None, data_device="gpu:1")
        self.assertIsInstance(res, list)
        for r in res:
            self.check_gpu(r, 1)

        res = paddle_move_data_to_device(paddle_list, device="cpu", data_device="gpu:1")
        self.assertIsInstance(res, list)
        for r in res:
            self.check_cpu(r)

        res = paddle_move_data_to_device(paddle_list, device="gpu:0", data_device=None)
        self.assertIsInstance(res, list)
        for r in res:
            self.check_gpu(r, 0)

        res = paddle_move_data_to_device(paddle_list, device="gpu:1", data_device="cpu")
        self.assertIsInstance(res, list)
        for r in res:
            self.check_gpu(r, 1)

    def test_tensor_tuple_transfer(self):
        """
        测试张量元组的迁移
        """

        paddle_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        paddle_tuple = tuple(paddle_list)
        res = paddle_move_data_to_device(paddle_tuple, device=None, data_device="gpu:1")
        self.assertIsInstance(res, tuple)
        for r in res:
            self.check_gpu(r, 1)

        res = paddle_move_data_to_device(paddle_tuple, device="cpu", data_device="gpu:1")
        self.assertIsInstance(res, tuple)
        for r in res:
            self.check_cpu(r)

        res = paddle_move_data_to_device(paddle_tuple, device="gpu:0", data_device=None)
        self.assertIsInstance(res, tuple)
        for r in res:
            self.check_gpu(r, 0)

        res = paddle_move_data_to_device(paddle_tuple, device="gpu:1", data_device="cpu")
        self.assertIsInstance(res, tuple)
        for r in res:
            self.check_gpu(r, 1)

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

        res = paddle_move_data_to_device(paddle_dict, device="gpu:0", data_device=None)
        self.assertIsInstance(res, dict)
        self.check_gpu(res["tensor"], 0)
        self.assertIsInstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 0)
        self.assertIsInstance(res["int"], int)
        self.assertIsInstance(res["string"], str)
        self.assertIsInstance(res["dict"], dict)
        self.assertIsInstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 0)
        self.check_gpu(res["dict"]["tensor"], 0)

        res = paddle_move_data_to_device(paddle_dict, device="gpu:0", data_device="cpu")
        self.assertIsInstance(res, dict)
        self.check_gpu(res["tensor"], 0)
        self.assertIsInstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 0)
        self.assertIsInstance(res["int"], int)
        self.assertIsInstance(res["string"], str)
        self.assertIsInstance(res["dict"], dict)
        self.assertIsInstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 0)
        self.check_gpu(res["dict"]["tensor"], 0)

        res = paddle_move_data_to_device(paddle_dict, device=None, data_device="gpu:1")
        self.assertIsInstance(res, dict)
        self.check_gpu(res["tensor"], 1)
        self.assertIsInstance(res["list"], list)
        for t in res["list"]:
            self.check_gpu(t, 1)
        self.assertIsInstance(res["int"], int)
        self.assertIsInstance(res["string"], str)
        self.assertIsInstance(res["dict"], dict)
        self.assertIsInstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_gpu(t, 1)
        self.check_gpu(res["dict"]["tensor"], 1)

        res = paddle_move_data_to_device(paddle_dict, device="cpu", data_device="gpu:0")
        self.assertIsInstance(res, dict)
        self.check_cpu(res["tensor"])
        self.assertIsInstance(res["list"], list)
        for t in res["list"]:
            self.check_cpu(t)
        self.assertIsInstance(res["int"], int)
        self.assertIsInstance(res["string"], str)
        self.assertIsInstance(res["dict"], dict)
        self.assertIsInstance(res["dict"]["list"], list)
        for t in res["dict"]["list"]:
            self.check_cpu(t)
        self.check_cpu(res["dict"]["tensor"])
