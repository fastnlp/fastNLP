import pytest

from fastNLP.core.drivers import TorchSingleDriver, TorchDDPDriver
from fastNLP.core.drivers.torch_driver.initialize_torch_driver import initialize_torch_driver
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch import device as torchdevice
else:
    from fastNLP.core.utils.dummy_class import DummyClass as torchdevice

@pytest.mark.torch
def test_incorrect_driver():

    model = TorchNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_torch_driver("paddle", 0, model)


@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    ["cpu", "cuda:0", 0, torchdevice("cuda:0")]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch"]
)
def test_get_single_device(driver, device):
    """
    测试正常情况下初始化TorchSingleDriver的情况
    """

    model = TorchNormalModel_Classification_1(20, 10)
    driver = initialize_torch_driver(driver, device, model)
    assert isinstance(driver, TorchSingleDriver)

@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    [[0, 1], -1]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch"]
)
@magic_argv_env_context
def test_get_ddp(driver, device):
    """
    测试 ddp 多卡的初始化情况
    """

    model = TorchNormalModel_Classification_1(20, 10)
    driver = initialize_torch_driver(driver, device, model)

    assert isinstance(driver, TorchDDPDriver)

@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    [-2, [0, 20, 3], [-2], 20]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch"]
)
def test_device_out_of_range(driver, device):
    """
    测试传入的device超过范围的情况
    """
    model = TorchNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_torch_driver(driver, device, model) 