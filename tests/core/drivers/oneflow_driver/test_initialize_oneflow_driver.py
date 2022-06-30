import pytest

from fastNLP.core.drivers import OneflowSingleDriver, OneflowDDPDriver
from fastNLP.core.drivers.oneflow_driver.initialize_oneflow_driver import initialize_oneflow_driver
from tests.helpers.models.oneflow_model import OneflowNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
if _NEED_IMPORT_ONEFLOW:
    import oneflow
    from oneflow import device as oneflowdevice
else:
    from fastNLP.core.utils.dummy_class import DummyClass as oneflowdevice

@pytest.mark.oneflow
def test_incorrect_driver():

    model = OneflowNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_oneflow_driver("paddle", 0, model)


@pytest.mark.oneflow
@pytest.mark.parametrize(
    "device", 
    ["cpu", "cuda:0", 0, oneflowdevice("cuda:0")]
)
@pytest.mark.parametrize(
    "driver", 
    ["oneflow"]
)
def test_get_single_device(driver, device):
    """
    测试正常情况下初始化OneflowSingleDriver的情况
    """

    model = OneflowNormalModel_Classification_1(20, 10)
    driver = initialize_oneflow_driver(driver, device, model)
    assert isinstance(driver, OneflowSingleDriver)

@pytest.mark.oneflow
@pytest.mark.parametrize(
    "device", 
    [[0, 1], -1]
)
@pytest.mark.parametrize(
    "driver", 
    ["oneflow"]
)
@magic_argv_env_context
def test_get_ddp(driver, device):
    """
    测试 ddp 多卡的初始化情况
    """

    model = OneflowNormalModel_Classification_1(20, 10)
    with pytest.raises(RuntimeError):
        driver = initialize_oneflow_driver(driver, device, model)

    # assert isinstance(driver, OneflowDDPDriver)

@pytest.mark.oneflow
@pytest.mark.parametrize(
    "device", 
    [-2, [0, 20, 3], [-2], 20]
)
@pytest.mark.parametrize(
    "driver", 
    ["oneflow"]
)
def test_device_out_of_range(driver, device):
    """
    测试传入的device超过范围的情况
    """
    model = OneflowNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_oneflow_driver(driver, device, model) 