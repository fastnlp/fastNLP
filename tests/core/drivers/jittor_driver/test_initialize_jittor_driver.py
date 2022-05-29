import pytest

from fastNLP.core.drivers import JittorSingleDriver, JittorMPIDriver
from fastNLP.core.drivers.jittor_driver.initialize_jittor_driver import initialize_jittor_driver
from tests.helpers.models.jittor_model import JittorNormalModel_Classification_1
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    import jittor as jt

@pytest.mark.jittor
def test_incorrect_driver():

    model =  JittorNormalModel_Classification_1(20, 10)
    with pytest.raises(ValueError):
        driver = initialize_jittor_driver("torch", 0, model)

@pytest.mark.jittor
@pytest.mark.parametrize(
    "device", 
    ["cpu", "gpu", None, "cuda"]
)
def test_get_single_device(device):
    """
    测试正常情况下初始化 JittorSingleDriver 的情况
    """

    model =  JittorNormalModel_Classification_1(20, 10)
    driver = initialize_jittor_driver("jittor", device, model)
    assert isinstance(driver, JittorSingleDriver)

@pytest.mark.jittor
@pytest.mark.parametrize(
    "device", 
    [[0, 2, 3], 1, 2]
)
def test_get_mpi(device):
    """
    测试 jittor 多卡的初始化情况
    """

    model =  JittorNormalModel_Classification_1(20, 10)
    with pytest.raises(NotImplementedError):
        driver = initialize_jittor_driver("jittor", device, model)

    # assert isinstance(driver, JittorMPIDriver)
