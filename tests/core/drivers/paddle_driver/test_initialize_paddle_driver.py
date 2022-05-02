import pytest

from fastNLP.core.drivers import PaddleSingleDriver, PaddleFleetDriver
from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
from fastNLP.envs import get_gpu_count
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context

import paddle

@pytest.mark.paddle
def test_incorrect_driver():

    model = PaddleNormalModel_Classification_1(2, 100)
    with pytest.raises(ValueError):
        driver = initialize_paddle_driver("torch", 0, model)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    ["cpu", "gpu:0", 0]
)
@pytest.mark.parametrize(
    "driver", 
    ["paddle"]
)
def test_get_single_device(driver, device):
    """
    测试正常情况下初始化 PaddleSingleDriver 的情况
    """

    model = PaddleNormalModel_Classification_1(2, 100)
    driver = initialize_paddle_driver(driver, device, model)
    assert isinstance(driver, PaddleSingleDriver)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    [0, 1, [1]]
)
@pytest.mark.parametrize(
    "driver", 
    ["fleet"]
)
@magic_argv_env_context
def test_get_fleet_2(driver, device):
    """
    测试 fleet 多卡的初始化情况，但传入了单个 gpu
    """

    model = PaddleNormalModel_Classification_1(64, 10)
    driver = initialize_paddle_driver(driver, device, model)

    assert isinstance(driver, PaddleFleetDriver)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    [[0, 2, 3], -1]
)
@pytest.mark.parametrize(
    "driver", 
    ["paddle", "fleet"]
)
@magic_argv_env_context
def test_get_fleet(driver, device):
    """
    测试 fleet 多卡的初始化情况
    """

    model = PaddleNormalModel_Classification_1(64, 10)
    driver = initialize_paddle_driver(driver, device, model)

    assert isinstance(driver, PaddleFleetDriver)

@pytest.mark.paddle
@pytest.mark.parametrize(
    ("driver", "device"), 
    [("fleet", "cpu")]
)
@magic_argv_env_context
def test_get_fleet_cpu(driver, device):
    """
    测试试图在 cpu 上初始化分布式训练的情况
    """
    model = PaddleNormalModel_Classification_1(64, 10)
    with pytest.raises(ValueError):
        driver = initialize_paddle_driver(driver, device, model)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    [-2, [0, get_gpu_count() + 1, 3], [-2], get_gpu_count() + 1]
)
@pytest.mark.parametrize(
    "driver", 
    ["paddle", "fleet"]
)
@magic_argv_env_context
def test_device_out_of_range(driver, device):
    """
    测试传入的device超过范围的情况
    """
    model = PaddleNormalModel_Classification_1(2, 100)
    with pytest.raises(ValueError):
        driver = initialize_paddle_driver(driver, device, model)
