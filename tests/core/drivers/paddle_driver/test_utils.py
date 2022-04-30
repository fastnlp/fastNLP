import pytest

from fastNLP.core.drivers.paddle_driver.utils import (
    get_device_from_visible,
    replace_batch_sampler,
    replace_sampler,
)
from fastNLP.core.samplers import RandomBatchSampler, RandomSampler

import paddle
from paddle.io import DataLoader, BatchSampler

from tests.helpers.datasets.paddle_data import PaddleNormalDataset

@pytest.mark.parametrize(
    ("user_visible_devices, cuda_visible_devices, device, output_type, correct"),
    (
        ("0,1,2,3,4,5,6,7", "0", "cpu", str, "cpu"),
        ("0,1,2,3,4,5,6,7", "0", "cpu", int, "cpu"),
        ("0,1,2,3,4,5,6,7", "3,4,5", "gpu:4", int, 1),
        ("0,1,2,3,4,5,6,7", "3,4,5", "gpu:5", str, "gpu:2"),
        ("3,4,5,6", "3,5", 0, int, 0),
        ("3,6,7,8", "6,7,8", "gpu:2", str, "gpu:1"),
    )
)
def test_get_device_from_visible_str(user_visible_devices, cuda_visible_devices, device, output_type, correct):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["USER_CUDA_VISIBLE_DEVICES"] = user_visible_devices
    res = get_device_from_visible(device, output_type)
    assert res == correct

def test_replace_batch_sampler():
    dataset = PaddleNormalDataset(10)
    dataloader = DataLoader(dataset, batch_size=32)
    batch_sampler = RandomBatchSampler(dataloader.batch_sampler, batch_size=16, drop_last=False)

    replaced_loader = replace_batch_sampler(dataloader, batch_sampler)

    assert not (replaced_loader is dataloader)
    assert isinstance(replaced_loader.batch_sampler, RandomBatchSampler)
    assert isinstance(replaced_loader.dataset, PaddleNormalDataset)
    assert len(replaced_loader.dataset) == len(dataset)
    assert replaced_loader.batch_sampler.batch_size == 16

def test_replace_sampler():
    dataset = PaddleNormalDataset(10)
    dataloader = DataLoader(dataset, batch_size=32)
    sampler = RandomSampler(dataset)

    replaced_loader = replace_sampler(dataloader, sampler)

    assert not (replaced_loader is dataloader)
    assert isinstance(replaced_loader.batch_sampler, BatchSampler)
    assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)