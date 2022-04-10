import pytest

from fastNLP.envs.set_backend import set_env
from fastNLP.envs.set_env_on_import import set_env_on_import_paddle

set_env_on_import_paddle()
set_env("paddle")
import paddle
from paddle.io import DataLoader, BatchSampler

from fastNLP.core.drivers.paddle_driver.single_device import PaddleSingleDriver
from fastNLP.core.samplers.reproducible_sampler import RandomSampler
from fastNLP.core.samplers import ReproducibleBatchSampler
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification
from tests.helpers.datasets.paddle_data import PaddleDataset_MNIST, PaddleRandomDataset
from fastNLP.core import synchronize_safe_rm


############################################################################
#
# 测试save和load相关的功能
#
############################################################################

def generate_random_driver(features, labels):
    """
    生成driver
    """
    model = PaddleNormalModel_Classification(labels, features)
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)
    driver = PaddleSingleDriver(model)
    driver.set_optimizers(opt)

    return driver

@pytest.fixture
def prepare_test_save_load():
    dataset = PaddleRandomDataset(num_of_data=320, features=64, labels=8)
    dataloader = DataLoader(dataset, batch_size=32)
    driver1, driver2 = generate_random_driver(64, 8), generate_random_driver(64, 8)
    return driver1, driver2, dataloader

def test_save_and_load(prepare_test_save_load):
    """
    测试save和load函数
    TODO optimizer的state_dict为空，暂时不测试
    """

    try:
        path = "model.pdparams"
        driver1, driver2, dataloader = prepare_test_save_load

        driver1.save(path, {})
        driver2.load(path)

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
        path = "model.pdparams"
        driver1, driver2, dataloader = prepare_test_save_load

        driver1.save_model(path)
        driver2.model.load_dict(driver2.load_model(path))

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
        path = "model.pdparams"
        driver1, driver2, dataloader = prepare_test_save_load

        driver1.save_model(path, only_state_dict=False, input_spec=[next(iter(dataloader))["x"]])
        driver2.model = driver2.load_model(path, load_dict=False)

        for batch in dataloader:
            res1 = driver1.validate_step(batch)
            res2 = driver2.validate_step(batch)

            assert paddle.equal_all(res1["pred"], res2["pred"])
    finally:
        synchronize_safe_rm(path)


class TestSingleDeviceFunction:
    """
    测试其它函数的测试例
    """

    @classmethod
    def setup_class(cls):
        model = PaddleNormalModel_Classification(10, 784)
        cls.driver = PaddleSingleDriver(model)

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
        self.driver = PaddleSingleDriver(PaddleNormalModel_Classification(10, 784), "cpu")
        device = self.driver.get_model_device()
        assert device == "cpu", device

    def test_get_model_device_gpu(self):
        """
        测试get_model_device
        """
        self.driver = PaddleSingleDriver(PaddleNormalModel_Classification(10, 784), "gpu:0")
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
        "dist_sampler",
        ["dist", ReproducibleBatchSampler(BatchSampler(PaddleDataset_MNIST("train")), 32, False), RandomSampler(PaddleDataset_MNIST("train"))]
    )
    @pytest.mark.parametrize(
        "reproducible",
        [True, False]
    )
    def test_repalce_sampler(self, dist_sampler, reproducible):
        """
        测试replace_sampler函数
        """
        dataloader = DataLoader(PaddleDataset_MNIST("train"), batch_size=100, shuffle=True)

        res = self.driver.set_dist_repro_dataloader(dataloader, dist_sampler, reproducible)