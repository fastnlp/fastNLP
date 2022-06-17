import os

import pytest

from fastNLP.core.drivers import PaddleSingleDriver, PaddleFleetDriver
from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
from fastNLP.envs import get_gpu_count
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
if _NEED_IMPORT_PADDLE:
    import paddle

@pytest.mark.paddle
def test_incorrect_driver():

    model = PaddleNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_paddle_driver("torch", 0, model)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    ["cpu", "gpu:0", 0]
)
def test_get_single_device(device):
    """
    测试正常情况下初始化 PaddleSingleDriver 的情况
    """

    model = PaddleNormalModel_Classification_1(20, 10)
    driver = initialize_paddle_driver("paddle", device, model)
    assert isinstance(driver, PaddleSingleDriver)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    [[0, 2, 3], -1]
)
@magic_argv_env_context
def test_get_fleet(device):
    """
    测试 fleet 多卡的初始化情况
    """
    flag = False
    if "USER_CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["USER_CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        flag = True
    model = PaddleNormalModel_Classification_1(20, 10)
    driver = initialize_paddle_driver("paddle", device, model)
    if flag:
        del os.environ["USER_CUDA_VISIBLE_DEVICES"]

    assert isinstance(driver, PaddleFleetDriver)

@pytest.mark.paddle
@pytest.mark.parametrize(
    "device", 
    [-2, [0, get_gpu_count() + 1, 3], [-2], get_gpu_count() + 1]
)
@magic_argv_env_context
def test_device_out_of_range(device):
    """
    测试传入的device超过范围的情况
    """
    model = PaddleNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_paddle_driver("paddle", device, model)
