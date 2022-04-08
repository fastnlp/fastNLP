import pytest

from fastNLP.envs.set_backend import set_env
from fastNLP.envs.set_env_on_import import set_env_on_import_paddle

set_env_on_import_paddle()
set_env("paddle")
import paddle

from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
from fastNLP.core.drivers.paddle_driver.single_device import PaddleSingleDriver
from fastNLP.core.drivers.paddle_driver.fleet import PaddleFleetDriver
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification


def test_incorrect_driver():

    with pytest.raises(ValueError):
        driver = initialize_paddle_driver("torch")

@pytest.mark.parametrize(
    "device", 
    ["cpu", "gpu:0", [1, 2, 3], 0, "gpu:1"]
)
def test_get_single_device(device):
    """
    测试正常情况下初始化PaddleSingleDriver的情况
    """

    model = PaddleNormalModel_Classification(2, 100)
    driver = initialize_paddle_driver("paddle", device, model)

    assert isinstance(driver, PaddleSingleDriver)

@pytest.mark.parametrize(
    "device", 
    ["cpu", "gpu:0", [1, 2, 3], 0, "gpu:1"]
)
def test_get_single_device_with_visiblde_devices(device):
    """
    测试 CUDA_VISIBLE_DEVICES 启动时初始化PaddleSingleDriver的情况
    """
    # TODO

    model = PaddleNormalModel_Classification(2, 100)
    driver = initialize_paddle_driver("paddle", device, model)

    assert isinstance(driver, PaddleSingleDriver)

@pytest.mark.parametrize(
    "device", 
    [[1, 2, 3]]
)
def test_get_fleet(device):
    """
    测试 fleet 多卡的初始化情况
    """

    model = PaddleNormalModel_Classification(2, 100)
    driver = initialize_paddle_driver("paddle", device, model)

    assert isinstance(driver, PaddleFleetDriver)

@pytest.mark.parametrize(
    "device", 
    [[1,2,3]]
)
def test_get_fleet(device):
    """
    测试 launch 启动 fleet 多卡的初始化情况
    """
    # TODO

    model = PaddleNormalModel_Classification(2, 100)
    driver = initialize_paddle_driver("paddle", device, model)

    assert isinstance(driver, PaddleFleetDriver)

def test_device_out_of_range(device):
    """
    测试传入的device超过范围的情况
    """
    pass