import os
import pytest
os.environ["FASTNLP_BACKEND"] = "paddle"

from fastNLP.core.drivers.paddle_driver.paddle_driver import PaddleDriver
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleNormalDataset
from tests.helpers.datasets.torch_data import TorchNormalDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

import torch
import paddle
from paddle.io import DataLoader

class TestPaddleDriverFunctions:
    """
    PaddleDriver的测试类，使用仅测试部分函数，其它的由PaddleSingleDriver和PaddleFleetDriver完成测试
    """

    @classmethod
    def setup_class(self):
        model = PaddleNormalModel_Classification_1(10, 32)
        self.driver = PaddleDriver(model)

    def test_check_single_optimizer_legality(self):
        """
        测试传入单个optimizer时的表现
        """
        optimizer = paddle.optimizer.Adam(
            parameters=self.driver.model.parameters(),
            learning_rate=0.01
        )

        self.driver.set_optimizers(optimizer)

        optimizer = torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        # 传入torch的optimizer时，应该报错ValueError
        with self.assertRaises(ValueError) as cm:
            self.driver.set_optimizers(optimizer)

    def test_check_optimizers_legality(self):
        """
        测试传入optimizer list的表现
        """
        optimizers = [
            paddle.optimizer.Adam(
                parameters=self.driver.model.parameters(),
                learning_rate=0.01
            ) for i in range(10)
        ]

        self.driver.set_optimizers(optimizers)

        optimizers += [
            torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        ]

        with self.assertRaises(ValueError) as cm:
            self.driver.set_optimizers(optimizers)

    def test_check_dataloader_legality_in_train(self):
        """
        测试is_train参数为True时，_check_dataloader_legality函数的表现
        """
        dataloader = paddle.io.DataLoader(PaddleNormalDataset())
        PaddleDriver._check_dataloader_legality(dataloader, "dataloader", True)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        with self.assertRaises(ValueError) as cm:
            PaddleDriver._check_dataloader_legality(dataloader, "dataloader", True)

        # 创建torch的dataloader
        dataloader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        with self.assertRaises(ValueError) as cm:
            PaddleDriver._check_dataloader_legality(dataloader, "dataloader", True)

    def test_check_dataloader_legacy_in_test(self):
        """
        测试is_train参数为False时，_check_dataloader_legality函数的表现
        """
        # 此时传入的应该是dict
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset())
        }
        PaddleDriver._check_dataloader_legality(dataloader, "dataloader", False)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        }
        PaddleDriver._check_dataloader_legality(dataloader, "dataloader", False)

        # 传入的不是dict，应该报错
        dataloader = paddle.io.DataLoader(PaddleNormalDataset())
        with self.assertRaises(ValueError) as cm:
            PaddleDriver._check_dataloader_legality(dataloader, "dataloader", False)

        # 创建torch的dataloader
        train_loader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        dataloader = {"train": train_loader, "test": test_loader}
        with self.assertRaises(ValueError) as cm:
            PaddleDriver._check_dataloader_legality(dataloader, "dataloader", False)

    def test_tensor_to_numeric(self):
        """
        测试tensor_to_numeric函数
        """
        # 单个张量
        tensor = paddle.to_tensor(3)
        res = PaddleDriver.tensor_to_numeric(tensor)
        self.assertEqual(res, 3)

        tensor = paddle.rand((3, 4))
        res = PaddleDriver.tensor_to_numeric(tensor)
        self.assertListEqual(res, tensor.tolist())

        # 张量list
        tensor_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        res = PaddleDriver.tensor_to_numeric(tensor_list)
        self.assertTrue(res, list)
        tensor_list = [t.tolist() for t in tensor_list]
        self.assertListEqual(res, tensor_list)

        # 张量tuple
        tensor_tuple = tuple([paddle.rand((6, 4, 2)) for i in range(10)])
        res = PaddleDriver.tensor_to_numeric(tensor_tuple)
        self.assertTrue(res, tuple)
        tensor_tuple = tuple([t.tolist() for t in tensor_tuple])
        self.assertTupleEqual(res, tensor_tuple)

        # 张量dict
        tensor_dict = {
            "tensor": paddle.rand((3, 4)),
            "list": [paddle.rand((6, 4, 2)) for i in range(10)],
            "dict":{
                "list": [paddle.rand((6, 4, 2)) for i in range(10)],
                "tensor": paddle.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }

        res = PaddleDriver.tensor_to_numeric(tensor_dict)
        self.assertIsInstance(res, dict)
        self.assertListEqual(res["tensor"], tensor_dict["tensor"].tolist())
        self.assertIsInstance(res["list"], list)
        for r, d in zip(res["list"], tensor_dict["list"]):
            self.assertListEqual(r, d.tolist())
        self.assertIsInstance(res["int"], int)
        self.assertIsInstance(res["string"], str)
        self.assertIsInstance(res["dict"], dict)
        self.assertIsInstance(res["dict"]["list"], list)
        for r, d in zip(res["dict"]["list"], tensor_dict["dict"]["list"]):
            self.assertListEqual(r, d.tolist())
        self.assertListEqual(res["dict"]["tensor"], tensor_dict["dict"]["tensor"].tolist())

    def test_set_model_mode(self):
        """
        测试set_model_mode函数
        """
        self.driver.set_model_mode("train")
        self.assertTrue(self.driver.model.training)
        self.driver.set_model_mode("eval")
        self.assertFalse(self.driver.model.training)
        # 应该报错
        with self.assertRaises(AssertionError) as cm:
            self.driver.set_model_mode("test")

    def test_move_model_to_device_cpu(self):
        """
        测试move_model_to_device函数
        """
        PaddleDriver.move_model_to_device(self.driver.model, "cpu")
        self.assertTrue(self.driver.model.fc1.weight.place.is_cpu_place())

    def test_move_model_to_device_gpu(self):
        """
        测试move_model_to_device函数
        """
        PaddleDriver.move_model_to_device(self.driver.model, "gpu:0")
        self.assertTrue(self.driver.model.fc1.weight.place.is_gpu_place())
        self.assertEqual(self.driver.model.fc1.weight.place.gpu_device_id(), 0)

    def test_worker_init_function(self):
        """
        测试worker_init_function
        """
        # 先确保不影响运行
        # TODO：正确性
        PaddleDriver.worker_init_function(0)

    def test_set_deterministic_dataloader(self):
        """
        测试set_deterministic_dataloader
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(PaddleNormalDataset())
        self.driver.set_deterministic_dataloader(dataloader)

    def test_set_sampler_epoch(self):
        """
        测试set_sampler_epoch
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(PaddleNormalDataset())
        self.driver.set_sampler_epoch(dataloader, 0)

    def test_get_dataloader_args(self):
        """
        测试get_dataloader_args
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(PaddleNormalDataset())
        res = PaddleDriver.get_dataloader_args(dataloader)