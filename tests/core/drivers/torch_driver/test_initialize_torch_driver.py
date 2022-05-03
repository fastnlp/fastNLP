import pytest

from fastNLP.core.drivers import TorchSingleDriver, TorchDDPDriver
from fastNLP.core.drivers.torch_driver.initialize_torch_driver import initialize_torch_driver
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch import device as torchdevice
else:
    from fastNLP.core.utils.dummy_class import DummyClass as torchdevice

@pytest.mark.torch
def test_incorrect_driver():

    model = TorchNormalModel_Classification_1(2, 100)
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

    model = TorchNormalModel_Classification_1(2, 100)
    driver = initialize_torch_driver(driver, device, model)
    assert isinstance(driver, TorchSingleDriver)


@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    [0, 1]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch_ddp"]
)
@magic_argv_env_context
def test_get_ddp_2(driver, device):
    """
    测试 ddp 多卡的初始化情况，但传入了单个 gpu
    """

    model = TorchNormalModel_Classification_1(64, 10)
    driver = initialize_torch_driver(driver, device, model)

    assert isinstance(driver, TorchDDPDriver)
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    [[0, 2, 3], -1]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch", "torch_ddp"]
)
@magic_argv_env_context
def test_get_ddp(driver, device):
    """
    测试 ddp 多卡的初始化情况
    """

    model = TorchNormalModel_Classification_1(64, 10)
    driver = initialize_torch_driver(driver, device, model)

    assert isinstance(driver, TorchDDPDriver)
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize(
    ("driver", "device"), 
    [("torch_ddp", "cpu")]
)
def test_get_ddp_cpu(driver, device):
    """
    测试试图在 cpu 上初始化分布式训练的情况
    """
    model = TorchNormalModel_Classification_1(64, 10)
    with pytest.raises(ValueError):
        driver = initialize_torch_driver(driver, device, model)


@pytest.mark.torch
@pytest.mark.parametrize(
    "device", 
    [-2, [0, 20, 3], [-2], 20]
)
@pytest.mark.parametrize(
    "driver", 
    ["torch", "torch_ddp"]
)
def test_device_out_of_range(driver, device):
    """
    测试传入的device超过范围的情况
    """
    model = TorchNormalModel_Classification_1(2, 100)
    with pytest.raises(ValueError):
        driver = initialize_torch_driver(driver, device, model) 