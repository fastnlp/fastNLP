import os
os.environ["FASTNLP_BACKEND"] = "paddle"
import pytest

from fastNLP.core.drivers.paddle_driver.single_device import PaddleSingleDriver
from fastNLP.core.samplers.reproducible_sampler import RandomSampler
from fastNLP.core.samplers import ReproducibleBatchSampler
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleRandomMaxDataset
from tests.helpers.datasets.torch_data import TorchNormalDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from fastNLP.core import synchronize_safe_rm

import paddle
from paddle.io import DataLoader, BatchSampler
import torch


############################################################################
#
# 测试save和load相关的功能
#
############################################################################

def generate_random_driver(features, labels):
    """
    生成driver
    """
    model = PaddleNormalModel_Classification_1(labels, features)
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)
    driver = PaddleSingleDriver(model, device="cpu")
    driver.set_optimizers(opt)

    return driver

@pytest.fixture
def prepare_test_save_load():
    dataset = PaddleRandomMaxDataset(320, 10)
    dataloader = DataLoader(dataset, batch_size=32)
    driver1, driver2 = generate_random_driver(10, 10), generate_random_driver(10, 10)
    return driver1, driver2, dataloader

@pytest.mark.parametrize("reproducible", [True, False])
@pytest.mark.parametrize("only_state_dict", [True, False])
def test_save_and_load(prepare_test_save_load, reproducible, only_state_dict):
    """
    测试save和load函数
    TODO optimizer的state_dict为空，暂时不测试
    """

    try:
        path = "model.ckp"
        driver1, driver2, dataloader = prepare_test_save_load
        dataloader = driver1.set_dist_repro_dataloader(dataloader, "dist", reproducible)

        driver1.save(path, {}, dataloader, only_state_dict, should_save_model=True)
        driver2.load(path, dataloader, only_state_dict, should_load_model=True)

        for batch in dataloader:
            res1 = driver1.validate_step(batch)
            res2 = driver2.validate_step(batch)

            assert paddle.equal_all(res1["pred"], res2["pred"])
    finally:
        synchronize_safe_rm(path)

def test_save_and_load_state_dict(prepare_test_save_load):
    """
    测试save和load函数
    TODO optimizer的state_dict为空，暂时不测试
    """
    try:
        path = "dict"
        driver1, driver2, dataloader = prepare_test_save_load

        driver1.save_model(path)
        driver2.load_model(path)

        for batch in dataloader:
            res1 = driver1.validate_step(batch)
            res2 = driver2.validate_step(batch)

            assert paddle.equal_all(res1["pred"], res2["pred"])
    finally:
        synchronize_safe_rm(path)

def test_save_and_load_whole_model(prepare_test_save_load):
    """
    测试save和load函数
    TODO optimizer的state_dict为空，暂时不测试
    """
    try:
        path = "model"
        driver1, driver2, dataloader = prepare_test_save_load

        driver1.save_model(path, only_state_dict=False, input_spec=[next(iter(dataloader))["x"]])
        driver2.load_model(path, only_state_dict=False)

        for batch in dataloader:
            res1 = driver1.validate_step(batch)
            res2 = driver2.validate_step(batch)

            assert paddle.equal_all(res1["pred"], res2["pred"])
    finally:
        synchronize_safe_rm(path + ".pdiparams")
        synchronize_safe_rm(path + ".pdiparams.info")
        synchronize_safe_rm(path + ".pdmodel")


class TestSingleDeviceFunction:
    """
    测试其它函数的测试例
    """

    @classmethod
    def setup_class(cls):
        model = PaddleNormalModel_Classification_1(10, 784)
        cls.driver = PaddleSingleDriver(model, device="gpu")

    def test_unwrap_model(self):
        """
        测试能否运行
        """
        res = self.driver.unwrap_model()

    def test_check_evaluator_mode(self):
        """
        这两个函数没有返回值和抛出异常，仅检查是否有import错误等影响运行的因素
        """
        self.driver.check_evaluator_mode("validate")
        self.driver.check_evaluator_mode("test")

    def test_get_model_device_cpu(self):
        """
        测试get_model_device
        """
        self.driver = PaddleSingleDriver(PaddleNormalModel_Classification_1(10, 784), "cpu")
        device = self.driver.get_model_device()
        assert device == "cpu", device

    def test_get_model_device_gpu(self):
        """
        测试get_model_device
        """
        self.driver = PaddleSingleDriver(PaddleNormalModel_Classification_1(10, 784), "gpu:0")
        device = self.driver.get_model_device()
        assert device == "gpu:0", device

    def test_is_distributed(self):
        assert self.driver.is_distributed() == False

    def test_move_data_to_device(self):
        """
        这个函数仅调用了paddle_move_data_to_device，测试例在tests/core/utils/test_paddle_utils.py中
        就不重复测试了
        """
        self.driver.move_data_to_device(paddle.rand((32, 64)))

    @pytest.mark.parametrize(
        "dist_sampler", [
            "dist",
            ReproducibleBatchSampler(BatchSampler(PaddleRandomMaxDataset(320, 10)), 32, False),
            RandomSampler(PaddleRandomMaxDataset(320, 10))
        ]
    )
    @pytest.mark.parametrize(
        "reproducible",
        [True, False]
    )
    def test_repalce_sampler(self, dist_sampler, reproducible):
        """
        测试set_dist_repro_dataloader函数
        """
        dataloader = DataLoader(PaddleRandomMaxDataset(320, 10), batch_size=100, shuffle=True)

        res = self.driver.set_dist_repro_dataloader(dataloader, dist_sampler, reproducible)

class TestPaddleDriverFunctions:
    """
    使用 PaddleSingleDriver 测试基类的函数
    """

    @classmethod
    def setup_class(self):
        model = PaddleNormalModel_Classification_1(10, 32)
        self.driver = PaddleSingleDriver(model, device="gpu")

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
        PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", True)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        with self.assertRaises(ValueError) as cm:
            PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", True)

        # 创建torch的dataloader
        dataloader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        with self.assertRaises(ValueError) as cm:
            PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", True)

    def test_check_dataloader_legacy_in_test(self):
        """
        测试is_train参数为False时，_check_dataloader_legality函数的表现
        """
        # 此时传入的应该是dict
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset())
        }
        PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", False)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        }
        PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", False)

        # 传入的不是dict，应该报错
        dataloader = paddle.io.DataLoader(PaddleNormalDataset())
        with self.assertRaises(ValueError) as cm:
            PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", False)

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
            PaddleSingleDriver._check_dataloader_legality(dataloader, "dataloader", False)

    def test_tensor_to_numeric(self):
        """
        测试tensor_to_numeric函数
        """
        # 单个张量
        tensor = paddle.to_tensor(3)
        res = PaddleSingleDriver.tensor_to_numeric(tensor)
        self.assertEqual(res, 3)

        tensor = paddle.rand((3, 4))
        res = PaddleSingleDriver.tensor_to_numeric(tensor)
        self.assertListEqual(res, tensor.tolist())

        # 张量list
        tensor_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        res = PaddleSingleDriver.tensor_to_numeric(tensor_list)
        self.assertTrue(res, list)
        tensor_list = [t.tolist() for t in tensor_list]
        self.assertListEqual(res, tensor_list)

        # 张量tuple
        tensor_tuple = tuple([paddle.rand((6, 4, 2)) for i in range(10)])
        res = PaddleSingleDriver.tensor_to_numeric(tensor_tuple)
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

        res = PaddleSingleDriver.tensor_to_numeric(tensor_dict)
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
        PaddleSingleDriver.move_model_to_device(self.driver.model, "cpu")
        self.assertTrue(self.driver.model.fc1.weight.place.is_cpu_place())

    def test_move_model_to_device_gpu(self):
        """
        测试move_model_to_device函数
        """
        PaddleSingleDriver.move_model_to_device(self.driver.model, "gpu:0")
        self.assertTrue(self.driver.model.fc1.weight.place.is_gpu_place())
        self.assertEqual(self.driver.model.fc1.weight.place.gpu_device_id(), 0)

    def test_worker_init_function(self):
        """
        测试worker_init_function
        """
        # 先确保不影响运行
        # TODO：正确性
        PaddleSingleDriver.worker_init_function(0)

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
        res = PaddleSingleDriver.get_dataloader_args(dataloader)